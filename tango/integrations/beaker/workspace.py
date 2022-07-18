import json
import random
import tempfile
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Generator, Iterable, Iterator, Optional, TypeVar, Union, cast
from urllib.parse import ParseResult

import petname
from beaker import Beaker, Dataset, DatasetConflict, DatasetNotFound, Digest

from tango.common.exceptions import StepStateError
from tango.common.logging import file_handler
from tango.common.util import exception_to_string, tango_cache_dir, utc_now_datetime
from tango.step import Step
from tango.step_cache import StepCache
from tango.step_info import StepInfo, StepState
from tango.workspace import Run, Workspace

from .common import (
    BeakerStepLock,
    Constants,
    dataset_url,
    run_dataset_name,
    step_dataset_name,
)
from .step_cache import BeakerStepCache

T = TypeVar("T")


@Workspace.register("beaker")
class BeakerWorkspace(Workspace):
    """
    This is a :class:`~tango.workspace.Workspace` that stores step artifacts on `Beaker`_.

    :param workspace: The name or ID of the Beaker workspace to use.
    :param kwargs: Additional keyword arguments passed to :meth:`Beaker.from_env() <beaker.Beaker.from_env()>`.

    .. tip::
        Registered as :class:`~tango.workspace.Workspace` under the name "beaker".
    """

    STEP_INFO_CACHE_SIZE = 512

    def __init__(self, beaker_workspace: str, **kwargs):
        super().__init__()
        self.beaker = Beaker.from_env(default_workspace=beaker_workspace, session=True, **kwargs)
        self.cache = BeakerStepCache(beaker=self.beaker)
        self.steps_dir = tango_cache_dir() / "beaker_workspace"
        self.locks: Dict[Step, BeakerStepLock] = {}
        self._step_info_cache: "OrderedDict[Digest, StepInfo]" = OrderedDict()

    @property
    def url(self) -> str:
        return f"beaker://{self.beaker.workspace.get().full_name}"

    @classmethod
    def from_parsed_url(cls, parsed_url: ParseResult) -> Workspace:
        workspace: str
        if parsed_url.netloc and parsed_url.path:
            # e.g. "beaker://ai2/my-workspace"
            workspace = parsed_url.netloc + parsed_url.path
        elif parsed_url.netloc:
            # e.g. "beaker://my-workspace"
            workspace = parsed_url.netloc
        else:
            raise ValueError(f"Bad URL for Beaker workspace '{parsed_url}'")
        return cls(workspace)

    @property
    def step_cache(self) -> StepCache:
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
        try:
            dataset = self.beaker.dataset.get(step_dataset_name(step_or_unique_id))
            file_info = self.beaker.dataset.file_info(dataset, Constants.STEP_INFO_FNAME)
            step_info: StepInfo
            if file_info.digest in self._step_info_cache:
                step_info = self._step_info_cache.pop(file_info.digest)
            else:
                step_info_bytes = b"".join(
                    self.beaker.dataset.stream_file(dataset, file_info, quiet=True)
                )
                step_info = StepInfo.from_json_dict(json.loads(step_info_bytes))
            self._step_info_cache[file_info.digest] = step_info
            while len(self._step_info_cache) > self.STEP_INFO_CACHE_SIZE:
                self._step_info_cache.popitem(last=False)
            return step_info
        except (DatasetNotFound, FileNotFoundError):
            if not isinstance(step_or_unique_id, Step):
                raise KeyError(step_or_unique_id)
            step_info = StepInfo.new_from_step(step_or_unique_id)
            self._update_step_info(step_info)
            return step_info

    def step_starting(self, step: Step) -> None:
        # We don't do anything with uncacheable steps.
        if not step.cache_results:
            return

        # Get local file lock + remote Beaker dataset lock.
        lock = BeakerStepLock(self.beaker, step)
        lock.acquire()
        self.locks[step] = lock

        step_info = self.step_info(step)
        if step_info.state not in {StepState.INCOMPLETE, StepState.FAILED, StepState.UNCACHEABLE}:
            raise StepStateError(
                step,
                step_info.state,
                context=f"If you are certain the step is not running somewhere else, delete the step "
                f"datasets at {dataset_url(self.beaker.workspace.url(), step_dataset_name(step))}",
            )

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
        step_info.result_location = self.beaker.dataset.url(step_dataset_name(step))
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

    def register_run(self, targets: Iterable[Step], name: Optional[str] = None) -> Run:
        all_steps = set(targets)
        for step in targets:
            all_steps |= step.recursive_dependencies

        # Create a Beaker dataset that represents this run. The dataset which just contain
        # a JSON file that maps step names to step unique IDs.
        run_dataset: Dataset
        if name is None:
            # Find a unique name to use.
            while True:
                name = petname.generate() + str(random.randint(0, 100))
                try:
                    run_dataset = self.beaker.dataset.create(
                        run_dataset_name(cast(str, name)), commit=False
                    )
                except DatasetConflict:
                    continue
                else:
                    break
        else:
            try:
                run_dataset = self.beaker.dataset.create(name, commit=False)
            except DatasetConflict:
                raise ValueError("Run name '{name}' is already in use")

        # Collect step info and add data to run dataset.
        steps: Dict[str, StepInfo] = {}
        run_data: Dict[str, str] = {}
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)

            for step in all_steps:
                if step.name is None:
                    continue

                step_info = self.step_info(step)
                steps[step.name] = step_info
                run_data[step.name] = step.unique_id

            with open(tmp_dir / Constants.RUN_DATA_FNAME, "w") as f:
                json.dump(run_data, f)

            self.beaker.dataset.sync(run_dataset, tmp_dir / Constants.RUN_DATA_FNAME, quiet=True)

        return Run(name=cast(str, name), steps=steps, start_date=run_dataset.created)

    def registered_runs(self) -> Dict[str, Run]:
        import concurrent.futures

        runs: Dict[str, Run] = {}

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=9, thread_name_prefix="BeakerWorkspace.registered_runs()-"
        ) as executor:
            run_futures = []
            for dataset in self.beaker.workspace.datasets(
                match=Constants.RUN_DATASET_PREFIX, results=False
            ):
                run_futures.append(executor.submit(self._get_run_from_dataset, dataset))
            for future in concurrent.futures.as_completed(run_futures):
                run = future.result()
                if run is not None:
                    runs[run.name] = run

        return runs

    def registered_run(self, name: str) -> Run:
        err_msg = f"Run '{name}' not found in workspace"

        try:
            dataset_for_run = self.beaker.dataset.get(run_dataset_name(name))
            # Make sure the run is in our workspace.
            if dataset_for_run.workspace_ref.id != self.beaker.workspace.get().id:
                raise DatasetNotFound
        except DatasetNotFound:
            raise KeyError(err_msg)

        run = self._get_run_from_dataset(dataset_for_run)
        if run is None:
            raise KeyError(err_msg)
        else:
            return run

    @contextmanager
    def capture_logs_for_run(self, name: str) -> Generator[None, None, None]:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            log_file = Path(tmp_dir_name) / "out.log"
            try:
                with file_handler(log_file):
                    yield None
            finally:
                run_dataset = run_dataset_name(name)
                self.beaker.dataset.sync(run_dataset, log_file, quiet=True)
                self.beaker.dataset.commit(run_dataset)

    def _get_run_from_dataset(self, dataset: Dataset) -> Optional[Run]:
        if dataset.name is None:
            return None
        if not dataset.name.startswith(Constants.RUN_DATASET_PREFIX):
            return None

        try:
            run_name = dataset.name[len(Constants.RUN_DATASET_PREFIX) :]
            steps: Dict[str, StepInfo] = {}
            steps_info_bytes = b"".join(
                self.beaker.dataset.stream_file(dataset, Constants.RUN_DATA_FNAME, quiet=True)
            )
            steps_info = json.loads(steps_info_bytes)
        except (DatasetNotFound, FileNotFoundError):
            return None

        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=9, thread_name_prefix="BeakerWorkspace._get_run_from_dataset()-"
        ) as executor:
            step_info_futures = []
            for step_name, unique_id in steps_info.items():
                step_info_futures.append(executor.submit(self.step_info, unique_id))
            for future in concurrent.futures.as_completed(step_info_futures):
                steps[step_name] = future.result()

        return Run(name=run_name, start_date=dataset.created, steps=steps)

    def _update_step_info(self, step_info: StepInfo):
        dataset_name = step_dataset_name(step_info)

        step_info_dataset: Dataset
        try:
            step_info_dataset = self.beaker.dataset.create(dataset_name, commit=False)
        except DatasetConflict:
            step_info_dataset = self.beaker.dataset.get(dataset_name)

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)

            # Save step info.
            step_info_path = tmp_dir / Constants.STEP_INFO_FNAME
            with open(step_info_path, "w") as f:
                json.dump(step_info.to_json_dict(), f)
            self.beaker.dataset.sync(step_info_dataset, step_info_path, quiet=True)
