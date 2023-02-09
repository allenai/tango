import json
import logging
import os
import random
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Type, TypeVar, Union, cast
from urllib.parse import ParseResult

import petname
from beaker import Dataset as BeakerDataset
from beaker import (
    DatasetConflict,
    DatasetNotFound,
    Digest,
    Experiment,
    ExperimentNotFound,
)

from tango.common.util import make_safe_filename, tango_cache_dir
from tango.integrations.beaker.common import (
    BeakerStepLock,
    Constants,
    dataset_url,
    get_client,
)
from tango.integrations.beaker.step_cache import BeakerStepCache
from tango.step import Step
from tango.step_info import StepInfo
from tango.workspace import Run, Workspace
from tango.workspaces.remote_workspace import RemoteWorkspace

T = TypeVar("T")
U = TypeVar("U", Run, StepInfo)

logger = logging.getLogger(__name__)


@Workspace.register("beaker")
class BeakerWorkspace(RemoteWorkspace):
    """
    This is a :class:`~tango.workspace.Workspace` that stores step artifacts on `Beaker`_.

    .. tip::
        Registered as a :class:`~tango.workspace.Workspace` under the name "beaker".

    :param workspace: The name or ID of the Beaker workspace to use.
    :param kwargs: Additional keyword arguments passed to :meth:`Beaker.from_env() <beaker.Beaker.from_env()>`.
    """

    STEP_INFO_CACHE_SIZE = 512
    Constants = Constants
    NUM_CONCURRENT_WORKERS = 9

    def __init__(self, workspace: str, max_workers: Optional[int] = None, **kwargs):
        self.beaker = get_client(beaker_workspace=workspace, **kwargs)
        self._cache = BeakerStepCache(beaker=self.beaker)
        self._locks: Dict[Step, BeakerStepLock] = {}
        super().__init__()
        self.max_workers = max_workers
        self._disk_cache_dir = tango_cache_dir() / "beaker_cache" / "_objects"
        self._mem_cache: "OrderedDict[Digest, Union[StepInfo, Run]]" = OrderedDict()

    @property
    def cache(self):
        return self._cache

    @property
    def locks(self):
        return self._locks

    @property
    def steps_dir_name(self):
        return "beaker_workspace"

    @property
    def url(self) -> str:
        return f"beaker://{self.beaker.workspace.get().full_name}"

    def _step_location(self, step: Step) -> str:
        return dataset_url(self.beaker, self.Constants.step_artifact_name(step))

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

    def _remote_lock(self, step: Step) -> BeakerStepLock:
        return BeakerStepLock(
            self.beaker, step, current_beaker_experiment=self.current_beaker_experiment
        )

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
            while len(self._mem_cache) > self.STEP_INFO_CACHE_SIZE:
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
        while len(self._mem_cache) > self.STEP_INFO_CACHE_SIZE:
            self._mem_cache.popitem(last=False)

    def step_info(self, step_or_unique_id: Union[Step, str]) -> StepInfo:
        try:
            dataset = self.beaker.dataset.get(self.Constants.step_artifact_name(step_or_unique_id))
            file_info = self.beaker.dataset.file_info(dataset, self.Constants.STEP_INFO_FNAME)
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
        except (DatasetNotFound, FileNotFoundError):
            if not isinstance(step_or_unique_id, Step):
                raise KeyError(step_or_unique_id)
            step_info = StepInfo.new_from_step(step_or_unique_id)
            self._update_step_info(step_info)
            return step_info

    def _save_run(
        self, steps: Dict[str, StepInfo], run_data: Dict[str, str], name: Optional[str] = None
    ) -> Run:
        # Create a remote dataset that represents this run. The dataset which just contain
        # a JSON file that maps step names to step unique IDs.
        run_dataset: BeakerDataset
        if name is None:
            # Find a unique name to use.
            while True:
                name = petname.generate() + str(random.randint(0, 100))
                try:
                    run_dataset = self.beaker.dataset.create(
                        self.Constants.run_artifact_name(cast(str, name)), commit=False
                    )
                except DatasetConflict:
                    continue
                else:
                    break
        else:
            try:
                run_dataset = self.beaker.dataset.create(
                    self.Constants.run_artifact_name(name), commit=False
                )
            except DatasetConflict:
                raise ValueError(f"Run name '{name}' is already in use")

        # Upload run data to dataset.
        # NOTE: We don't commit the dataset here since we'll need to upload the logs file
        # after the run.
        self.beaker.dataset.upload(
            run_dataset, json.dumps(run_data).encode(), self.Constants.RUN_DATA_FNAME, quiet=True
        )

        return Run(name=cast(str, name), steps=steps, start_date=run_dataset.created)

    def registered_runs(self) -> Dict[str, Run]:
        import concurrent.futures

        runs: Dict[str, Run] = {}

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.NUM_CONCURRENT_WORKERS,
            thread_name_prefix="BeakerWorkspace.registered_runs()-",
        ) as executor:
            run_futures = []
            for dataset in self.beaker.workspace.iter_datasets(
                match=self.Constants.RUN_ARTIFACT_PREFIX, uncommitted=True, results=False
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
            dataset_for_run = self.beaker.dataset.get(self.Constants.run_artifact_name(name))
            # Make sure the run is in our workspace.
            if dataset_for_run.workspace_ref.id != self.beaker.workspace.get().id:  # type: ignore # TODO
                raise DatasetNotFound
        except DatasetNotFound:
            raise KeyError(err_msg)

        run = self._get_run_from_dataset(dataset_for_run)
        if run is None:
            raise KeyError(err_msg)
        else:
            return run

    def _save_run_log(self, name: str, log_file: Path):
        run_dataset = self.Constants.run_artifact_name(name)
        self.beaker.dataset.sync(run_dataset, log_file, quiet=True)
        self.beaker.dataset.commit(run_dataset)

    def _get_run_from_dataset(self, dataset: BeakerDataset) -> Optional[Run]:
        if dataset.name is None:
            return None
        if not dataset.name.startswith(self.Constants.RUN_ARTIFACT_PREFIX):
            return None

        try:
            run_name = dataset.name[len(self.Constants.RUN_ARTIFACT_PREFIX) :]
            steps_info_bytes = self.beaker.dataset.get_file(
                dataset, self.Constants.RUN_DATA_FNAME, quiet=True
            )
            steps_info = json.loads(steps_info_bytes)
        except (DatasetNotFound, FileNotFoundError):
            return None

        steps: Dict[str, StepInfo] = {}
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.NUM_CONCURRENT_WORKERS,
            thread_name_prefix="BeakerWorkspace._get_run_from_dataset()-",
        ) as executor:
            step_info_futures = []
            for unique_id in steps_info.values():
                step_info_futures.append(executor.submit(self.step_info, unique_id))
            for future in concurrent.futures.as_completed(step_info_futures):
                step_info = future.result()
                assert step_info.step_name is not None
                steps[step_info.step_name] = step_info

        return Run(name=run_name, start_date=dataset.created, steps=steps)

    def _update_step_info(self, step_info: StepInfo):
        dataset_name = self.Constants.step_artifact_name(step_info)

        step_info_dataset: BeakerDataset
        try:
            self.beaker.dataset.create(dataset_name, commit=False)
        except DatasetConflict:
            pass
        step_info_dataset = self.beaker.dataset.get(dataset_name)

        self.beaker.dataset.upload(
            step_info_dataset,  # folder name
            json.dumps(step_info.to_json_dict()).encode(),  # step info dict.
            self.Constants.STEP_INFO_FNAME,  # step info filename
            quiet=True,
        )
