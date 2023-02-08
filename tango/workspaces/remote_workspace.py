import logging
import tempfile
import warnings
from abc import abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Generator, Iterable, Iterator, Optional, Tuple, TypeVar, Union
from urllib.parse import ParseResult

from tango.common.exceptions import StepStateError
from tango.common.logging import file_handler
from tango.common.remote_utils import RemoteConstants
from tango.common.util import exception_to_string, tango_cache_dir, utc_now_datetime
from tango.step import Step
from tango.step_caches.remote_step_cache import RemoteStepCache
from tango.step_info import StepInfo, StepState
from tango.workspace import Run, Workspace

T = TypeVar("T")

logger = logging.getLogger(__name__)


class RemoteWorkspace(Workspace):
    """
    This is a :class:`~tango.workspace.Workspace` that stores step artifacts on some remote storage location.

    .. tip::
        All remote workspaces inherit from this.
    """

    Constants = RemoteConstants
    NUM_CONCURRENT_WORKERS: int = 9

    @property
    @abstractmethod
    def cache(self) -> RemoteStepCache:
        raise NotImplementedError()

    @property
    @abstractmethod
    def steps_dir_name(self) -> str:
        raise NotImplementedError()

    @property
    @abstractmethod
    def locks(self) -> Dict:
        raise NotImplementedError()

    @property
    def steps_dir(self) -> Path:
        return tango_cache_dir() / self.steps_dir_name

    @property
    @abstractmethod
    def url(self) -> str:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def from_parsed_url(cls, parsed_url: ParseResult) -> Workspace:
        raise NotImplementedError()

    @property
    def step_cache(self) -> RemoteStepCache:
        return self.cache

    def step_dir(self, step_or_unique_id: Union[Step, str]) -> Path:
        unique_id = (
            step_or_unique_id if isinstance(step_or_unique_id, str) else step_or_unique_id.unique_id
        )
        path = self.steps_dir / unique_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def work_dir(self, step: Step) -> Path:
        path = self.step_dir(step) / "work"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def step_info(self, step_or_unique_id: Union[Step, str]) -> StepInfo:
        raise NotImplementedError()

    @abstractmethod
    def _remote_lock(self, step: Step):
        raise NotImplementedError()

    @abstractmethod
    def _step_location(self, step: Step) -> str:
        raise NotImplementedError()

    def step_starting(self, step: Step) -> None:
        # We don't do anything with uncacheable steps.
        if not step.cache_results:
            return

        # Get local file lock + remote dataset lock.
        lock = self._remote_lock(step)
        lock.acquire()
        self.locks[step] = lock

        step_info = self.step_info(step)
        if step_info.state == StepState.RUNNING:
            # Since we've acquired the step lock we know this step can't be running
            # elsewhere. But the step state can still say its running if the last
            warnings.warn(
                f"Step info for step '{step.unique_id}' is invalid - says step is running "
                "although it shouldn't be. Ignoring and overwriting step start time.",
                UserWarning,
            )
        elif step_info.state not in {StepState.INCOMPLETE, StepState.FAILED, StepState.UNCACHEABLE}:
            self.locks.pop(step).release()
            raise StepStateError(
                step,
                step_info.state,
                context=f"If you are certain the step is not running somewhere else, delete the step "
                f"datasets at {self._step_location(step)}",
            )

        if step_info.state == StepState.FAILED:
            # Refresh the environment metadata since it might be out-of-date now.
            step_info.refresh()

        # Update StepInfo to mark as running.
        try:
            step_info.start_time = utc_now_datetime()
            step_info.end_time = None
            step_info.error = None
            step_info.result_location = None
            self._update_step_info(step_info)
        except:  # noqa: E722
            self.locks.pop(step).release()
            raise

    def step_finished(self, step: Step, result: T) -> T:
        # We don't do anything with uncacheable steps.
        if not step.cache_results:
            return result

        step_info = self.step_info(step)
        if step_info.state != StepState.RUNNING:
            raise StepStateError(step, step_info.state)

        # Update step info and save step execution metadata.
        # This needs to be done *before* adding the result to the cache, since adding
        # the result to the cache will commit the step dataset, making it immutable.
        step_info.end_time = utc_now_datetime()
        step_info.result_location = self._step_location(step)
        self._update_step_info(step_info)

        self.cache[step] = result
        if hasattr(result, "__next__"):
            assert isinstance(result, Iterator)
            # Caching the iterator will consume it, so we write it to the cache and then read from the cache
            # for the return value.
            result = self.cache[step]

        self.locks.pop(step).release()

        return result

    def step_failed(self, step: Step, e: BaseException) -> None:
        # We don't do anything with uncacheable steps.
        if not step.cache_results:
            return

        try:
            step_info = self.step_info(step)
            if step_info.state != StepState.RUNNING:
                raise StepStateError(step, step_info.state)
            step_info.end_time = utc_now_datetime()
            step_info.error = exception_to_string(e)
            self._update_step_info(step_info)
        finally:
            self.locks.pop(step).release()

    def _get_run_step_info(self, targets: Iterable[Step]) -> Tuple[Dict, Dict]:
        import concurrent.futures

        all_steps = set(targets)
        for step in targets:
            all_steps |= step.recursive_dependencies

        steps: Dict[str, StepInfo] = {}
        run_data: Dict[str, str] = {}

        # Collect step info.
        with concurrent.futures.ThreadPoolExecutor(
            thread_name_prefix="RemoteWorkspace._get_run_step_info()-"
        ) as executor:
            step_info_futures = []
            for step in all_steps:
                if step.name is None:
                    continue
                step_info_futures.append(executor.submit(self.step_info, step))
            for future in concurrent.futures.as_completed(step_info_futures):
                step_info = future.result()
                assert step_info.step_name is not None
                steps[step_info.step_name] = step_info
                run_data[step_info.step_name] = step_info.unique_id

        return steps, run_data

    @abstractmethod
    def _save_run(
        self, steps: Dict[str, StepInfo], run_data: Dict[str, str], name: Optional[str] = None
    ) -> Run:
        raise NotImplementedError()

    def register_run(self, targets: Iterable[Step], name: Optional[str] = None) -> Run:
        steps, run_data = self._get_run_step_info(targets)
        run = self._save_run(steps, run_data, name)
        return run

    @abstractmethod
    def _save_run_log(self, name: str, log_file: Path):
        raise NotImplementedError()

    @contextmanager
    def capture_logs_for_run(self, name: str) -> Generator[None, None, None]:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            log_file = Path(tmp_dir_name) / "out.log"
            try:
                with file_handler(log_file):
                    yield None
            finally:
                self._save_run_log(name, log_file)

    @abstractmethod
    def _update_step_info(self, step_info: StepInfo):
        raise NotImplementedError()
