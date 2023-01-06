import json
import logging
import os
import random
import tempfile
import warnings
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import (
    Dict,
    Generator,
    Iterable,
    Iterator,
    Optional,
    Type,
    TypeVar,
    Union,
    cast,
)
from urllib.parse import ParseResult

import petname
from beaker import (
    Dataset,
    DatasetConflict,
    DatasetNotFound,
    DatasetSort,
    Digest,
    Experiment,
    ExperimentNotFound,
)

from tango.common.exceptions import StepStateError
from tango.common.logging import file_handler
from tango.common.util import (
    exception_to_string,
    make_safe_filename,
    tango_cache_dir,
    utc_now_datetime,
)
from tango.step import Step
from tango.step_cache import StepCache
from tango.step_info import StepInfo, StepState
from tango.workspace import Run, RunSort, StepInfoSort, Workspace

from .common import (
    BeakerStepLock,
    Constants,
    dataset_url,
    get_client,
    run_dataset_name,
    step_dataset_name,
)
from .step_cache import BeakerStepCache

T = TypeVar("T")
U = TypeVar("U", Run, StepInfo)

logger = logging.getLogger(__name__)


@Workspace.register("beaker")
class BeakerWorkspace(Workspace):
    """
    This is a :class:`~tango.workspace.Workspace` that stores step artifacts on `Beaker`_.

    .. tip::
        Registered as a :class:`~tango.workspace.Workspace` under the name "beaker".

    :param beaker_workspace: The name or ID of the Beaker workspace to use.
    :param kwargs: Additional keyword arguments passed to :meth:`Beaker.from_env() <beaker.Beaker.from_env()>`.
    """

    MEM_CACHE_SIZE = 512

    def __init__(self, beaker_workspace: str, max_workers: Optional[int] = None, **kwargs):
        super().__init__()
        self.beaker = get_client(beaker_workspace=beaker_workspace, **kwargs)
        self.cache = BeakerStepCache(beaker=self.beaker)
        self.steps_dir = tango_cache_dir() / "beaker_workspace"
        self.locks: Dict[Step, BeakerStepLock] = {}
        self.max_workers = max_workers
        self._disk_cache_dir = tango_cache_dir() / "beaker_cache" / "_objects"
        self._mem_cache: "OrderedDict[Digest, Union[StepInfo, Run]]" = OrderedDict()

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

    @property
    def current_beaker_experiment(self) -> Optional[Experiment]:
        """
        When the workspace is being used within a Beaker experiment that was submitted
        by the Beaker executor, this will return the `Experiment` object.
        """
        experiment_name = os.environ.get("BEAKER_EXPERIMENT_NAME")
        if experiment_name is not None:
            try:
                return self.beaker.experiment.get(experiment_name)
            except ExperimentNotFound:
                return None
        else:
            return None

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

    def _get_object_from_cache(self, digest: Digest, o_type: Type[U]) -> Optional[U]:
        cache_path = self._disk_cache_dir / make_safe_filename(str(digest))
        if digest in self._mem_cache:
            cached = self._mem_cache.pop(digest)
            # Move to end.
            self._mem_cache[digest] = cached
            return cached if isinstance(cached, o_type) else None
        elif cache_path.is_file():
            try:
                with cache_path.open("r+t") as f:
                    json_dict = json.load(f)
                    cached = o_type.from_json_dict(json_dict)
            except Exception as exc:
                logger.warning("Error while loading object from workspace cache: %s", str(exc))
                try:
                    os.remove(cache_path)
                except FileNotFoundError:
                    pass
                return None
            # Add to in-memory cache.
            self._mem_cache[digest] = cached
            while len(self._mem_cache) > self.MEM_CACHE_SIZE:
                self._mem_cache.popitem(last=False)
            return cached  # type: ignore
        else:
            return None

    def _add_object_to_cache(self, digest: Digest, o: U):
        self._disk_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self._disk_cache_dir / make_safe_filename(str(digest))
        self._mem_cache[digest] = o
        with cache_path.open("w+t") as f:
            json.dump(o.to_json_dict(), f)
        while len(self._mem_cache) > self.MEM_CACHE_SIZE:
            self._mem_cache.popitem(last=False)

    def step_info(self, step_or_unique_id: Union[Step, str]) -> StepInfo:
        try:
            dataset = self.beaker.dataset.get(step_dataset_name(step_or_unique_id))
            return self._get_step_info_from_dataset(dataset)
        except (DatasetNotFound, FileNotFoundError):
            if not isinstance(step_or_unique_id, Step):
                raise KeyError(step_or_unique_id)
            step_info = StepInfo.new_from_step(step_or_unique_id)
            self._update_step_info(step_info)
            return step_info

    def _get_step_info_from_dataset(self, dataset: Dataset) -> StepInfo:
        file_info = self.beaker.dataset.file_info(dataset, Constants.STEP_INFO_FNAME)
        step_info: StepInfo
        cached = (
            None
            if file_info.digest is None
            else self._get_object_from_cache(file_info.digest, StepInfo)
        )
        if cached is not None:
            step_info = cached
        else:
            step_info_bytes = self.beaker.dataset.get_file(dataset, file_info, quiet=True)
            step_info = StepInfo.from_json_dict(json.loads(step_info_bytes))
            if file_info.digest is not None:
                self._add_object_to_cache(file_info.digest, step_info)
        return step_info

    def step_starting(self, step: Step) -> None:
        # We don't do anything with uncacheable steps.
        if not step.cache_results:
            return

        # Get local file lock + remote Beaker dataset lock.
        lock = BeakerStepLock(
            self.beaker, step, current_beaker_experiment=self.current_beaker_experiment
        )
        lock.acquire()
        self.locks[step] = lock

        step_info = self.step_info(step)
        if step_info.state == StepState.RUNNING:
            # Since we've acquired the step lock we know this step can't be running
            # elsewhere. But the step state can still say its running if the last
            # run exited before this workspace had a chance to update the step info.
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
                f"datasets at {dataset_url(self.beaker.workspace.url(), step_dataset_name(step))}",
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
        import concurrent.futures

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
                run_dataset = self.beaker.dataset.create(run_dataset_name(name), commit=False)
            except DatasetConflict:
                raise ValueError(f"Run name '{name}' is already in use")

        steps: Dict[str, StepInfo] = {}
        run_data: Dict[str, str] = {}

        # Collect step info.
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers, thread_name_prefix="BeakerWorkspace.register_run()-"
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

        # Upload run data to dataset.
        # NOTE: We don't commit the dataset here since we'll need to upload the logs file
        # after the run.
        self.beaker.dataset.upload(
            run_dataset, json.dumps(run_data).encode(), Constants.RUN_DATA_FNAME, quiet=True
        )

        return Run(name=cast(str, name), steps=steps, start_date=run_dataset.created)

    def search_registered_runs(
        self,
        *,
        sort_by: RunSort = RunSort.START_DATE,
        sort_descending: bool = True,
        match: Optional[str] = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
    ) -> Generator[Run, None, None]:
        if match is None:
            match = Constants.RUN_DATASET_PREFIX
        else:
            match = Constants.RUN_DATASET_PREFIX + match
        if sort_by == RunSort.START_DATE:
            sort = DatasetSort.created
        elif sort_by == RunSort.NAME:
            sort = DatasetSort.dataset_name
        else:
            raise NotImplementedError
        for dataset in self.beaker.workspace.iter_datasets(
            match=match,
            results=False,
            cursor=start or 0,
            limit=None if stop is None else stop - (start or 0),
            sort_by=sort,
            descending=sort_descending,
        ):
            run = self._get_run_from_dataset(dataset)
            if run is not None:
                yield run

    def search_step_info(
        self,
        *,
        sort_by: StepInfoSort = StepInfoSort.CREATED,
        sort_descending: bool = True,
        match: Optional[str] = None,
        start: int = 0,
        stop: Optional[int] = None,
    ) -> Generator[StepInfo, None, None]:
        if match is None:
            match = Constants.STEP_DATASET_PREFIX
        else:
            match = Constants.STEP_DATASET_PREFIX + match

        if sort_by == StepInfoSort.CREATED:
            sort = DatasetSort.created
        elif sort_by == StepInfoSort.UNIQUE_ID:
            sort = DatasetSort.dataset_name
        else:
            raise NotImplementedError

        for dataset in self.beaker.workspace.iter_datasets(
            match=match,
            results=False,
            cursor=start or 0,
            limit=None if stop is None else stop - (start or 0),
            sort_by=sort,
            descending=sort_descending,
        ):
            try:
                yield self._get_step_info_from_dataset(dataset)
            except (DatasetNotFound, FileNotFoundError):
                continue

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

        run_name = dataset.name[len(Constants.RUN_DATASET_PREFIX) :]

        try:
            file_info = self.beaker.dataset.file_info(dataset, Constants.RUN_DATA_FNAME)
            cached = (
                None
                if file_info.digest is None
                else self._get_object_from_cache(file_info.digest, Run)
            )
            if cached is not None:
                return cached

            steps_info_bytes = self.beaker.dataset.get_file(dataset, file_info, quiet=True)
            steps_info = json.loads(steps_info_bytes)
        except (DatasetNotFound, FileNotFoundError):
            return None

        import concurrent.futures

        steps: Dict[str, StepInfo] = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="BeakerWorkspace._get_run_from_dataset()-",
        ) as executor:
            step_info_futures = []
            for unique_id in steps_info.values():
                step_info_futures.append(executor.submit(self.step_info, unique_id))
            for future in concurrent.futures.as_completed(step_info_futures):
                step_info = future.result()
                assert step_info.step_name is not None
                steps[step_info.step_name] = step_info

        run = Run(name=run_name, start_date=dataset.created, steps=steps)
        if file_info.digest is not None:
            self._add_object_to_cache(file_info.digest, run)
        return run

    def _update_step_info(self, step_info: StepInfo):
        dataset_name = step_dataset_name(step_info)

        step_info_dataset: Dataset
        try:
            step_info_dataset = self.beaker.dataset.create(dataset_name, commit=False)
        except DatasetConflict:
            step_info_dataset = self.beaker.dataset.get(dataset_name)

        self.beaker.dataset.upload(
            step_info_dataset,
            json.dumps(step_info.to_json_dict()).encode(),
            Constants.STEP_INFO_FNAME,
            quiet=True,
        )
