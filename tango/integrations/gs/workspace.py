import datetime
import json
import random
from collections import OrderedDict
from typing import Dict, Iterable, Optional, TypeVar, cast
from urllib.parse import ParseResult

import petname
from google.cloud import datastore

from tango.common.remote_utils import RemoteDataset, RemoteDatasetConflict
from tango.integrations.gs.common import Constants, GCSStepLock, get_client
from tango.integrations.gs.step_cache import GSStepCache
from tango.step import Step
from tango.step_info import StepInfo
from tango.workspace import Run, Workspace
from tango.workspaces.remote_workspace import RemoteWorkspace

T = TypeVar("T")


@Workspace.register("gs")
class GSWorkspace(RemoteWorkspace):
    """
    This is a :class:`~tango.workspace.Workspace` that stores step artifacts on Google Cloud Storage.

    .. tip::
        Registered as a :class:`~tango.workspace.Workspace` under the name "gs".

    :param workspace: The name or ID of the Beaker workspace to use.
    :param kwargs: Additional keyword arguments passed to :meth:`GCSFileSystem() <gcsfs.GCSFileSystem()>`.

    .. important::
        You can use your default google cloud credentials by running `gcloud auth application-default login`.
        Otherwise, you can specify the credentials using `token` keyword argument.
    """

    Constants = Constants
    NUM_CONCURRENT_WORKERS = 9  # TODO: increase and check

    def __init__(self, workspace: str, **kwargs):
        self._client = get_client(gcs_workspace=workspace, **kwargs)
        self._cache = GSStepCache(workspace, client=self._client)
        self._locks: Dict[Step, GCSStepLock] = {}

        self._step_info_cache: "OrderedDict[str, StepInfo]" = OrderedDict()
        super().__init__()

        # TODO: Ugly. Fix.
        self._ds = datastore.Client(
            namespace=workspace, credentials=self._client.storage._credentials
        )

    @property
    def client(self):
        return self._client

    @property
    def cache(self):
        return self._cache

    @property
    def locks(self):
        return self._locks

    @property
    def steps_dir_name(self):
        return "gs_workspace"

    @classmethod
    def from_parsed_url(cls, parsed_url: ParseResult) -> Workspace:
        workspace: str
        if parsed_url.netloc and parsed_url.path:
            # e.g. "gs://ai2/my-workspace"
            workspace = parsed_url.netloc + parsed_url.path
        elif parsed_url.netloc:
            # e.g. "gs://my-workspace"
            workspace = parsed_url.netloc
        else:
            raise ValueError(f"Bad URL for GS workspace '{parsed_url}'")
        return cls(workspace)

    @property
    def url(self) -> str:
        return self.client.url()

    def _remote_lock(self, step: Step) -> GCSStepLock:
        return GCSStepLock(self.client, step)

    def register_run(self, targets: Iterable[Step], name: Optional[str] = None) -> Run:
        import concurrent.futures

        all_steps = set(targets)
        for step in targets:
            all_steps |= step.recursive_dependencies

        steps: Dict[str, StepInfo] = {}
        run_data: Dict[str, str] = {}

        # Collect step info.
        with concurrent.futures.ThreadPoolExecutor(
            thread_name_prefix="GSWorkspace.register_run()-"
        ) as executor:
            step_info_futures = []
            # TODO: explore getting all step_info objects at once
            for step in all_steps:
                if step.name is None:
                    continue
                step_info_futures.append(executor.submit(self.step_info, step))
            for future in concurrent.futures.as_completed(step_info_futures):
                step_info = future.result()
                assert step_info.step_name is not None
                steps[step_info.step_name] = step_info
                run_data[step_info.step_name] = step_info.unique_id

        if name is None:
            while True:
                name = petname.generate() + str(random.randint(0, 100))
                if not self._ds.get(self._ds.key("run", name)):
                    break
        else:
            if self._ds.get(self._ds.key("run", name)):
                raise ValueError(f"Run name '{name}' is already in use")

        run_entity = self._ds.entity(key=self._ds.key("run", name))
        run_entity["start_date"] = datetime.datetime.now()
        run_entity["steps"] = json.dumps(run_data).encode()
        self._ds.put(run_entity)

        return Run(name=cast(str, name), steps=steps, start_date=run_entity["start_date"])

    def _get_run_from_entity(self, run_entity: datastore.Entity) -> Optional[Run]:
        try:
            steps_info_bytes = run_entity["steps"]
            steps_info = json.loads(steps_info_bytes)
        except KeyError:
            return None

        import concurrent.futures

        steps: Dict[str, StepInfo] = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.NUM_CONCURRENT_WORKERS,
            thread_name_prefix="GSWorkspace._get_run_from_dataset()-",
        ) as executor:
            step_info_futures = []
            for unique_id in steps_info.values():
                step_info_futures.append(executor.submit(self.step_info, unique_id))
            for future in concurrent.futures.as_completed(step_info_futures):
                step_info = future.result()
                assert step_info.step_name is not None
                steps[step_info.step_name] = step_info

        return Run(name=run_entity.key.name, start_date=run_entity["start_date"], steps=steps)

    def registered_runs(self) -> Dict[str, Run]:
        import concurrent.futures

        runs: Dict[str, Run] = {}

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.NUM_CONCURRENT_WORKERS,
            thread_name_prefix="GSWorkspace.registered_runs()-",
        ) as executor:
            run_futures = []
            for run_entity in self._ds.query(kind="run").fetch():
                run_futures.append(executor.submit(self._get_run_from_entity, run_entity))
            for future in concurrent.futures.as_completed(run_futures):
                run = future.result()
                if run is not None:
                    runs[run.name] = run

        return runs

    def registered_run(self, name: str) -> Run:
        err_msg = f"Run '{name}' not found in workspace"

        run_entity = self._ds.get(key=self._ds.key("run", name))
        if not run_entity:
            raise KeyError(err_msg)

        run = self._get_run_from_entity(run_entity)
        if run is None:
            raise KeyError(err_msg)
        else:
            return run

    def _update_step_info(self, step_info: StepInfo):

        # TODO: this method needs to be atomic. we are updating two different places.

        step_info_entity = self._ds.entity(key=self._ds.key("stepinfo", step_info.unique_id))

        # We can store each key separately, but we only store things that are useful for querying.
        # TODO: do we want any other step_info keys?

        step_info_entity["step_name"] = step_info.step_name
        step_info_entity["start_time"] = step_info.start_time
        step_info_entity["end_time"] = step_info.end_time
        step_info_entity["result_location"] = step_info.result_location

        self._ds.put(step_info_entity)

        # The value of any property in datastore cannot be longer than 1500 bytes so we dump
        # the rest of the contents into the step_info.json file.
        dataset_name = self.Constants.step_dataset_name(step_info)
        step_info_dataset: RemoteDataset
        try:
            self.client.create(dataset_name)
        except RemoteDatasetConflict:
            pass

        step_info_dataset = self.client.get(dataset_name)
        self.client.upload(
            step_info_dataset,  # folder name
            json.dumps(step_info.to_json_dict()).encode(),  # step info dict.
            self.Constants.STEP_INFO_FNAME,  # step info filename
        )
