import datetime
import json
import random
import tempfile
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Generator, Iterable, Optional, TypeVar, Union, cast
from urllib.parse import ParseResult

import petname
from google.cloud import datastore

from tango.common.logging import file_handler
from tango.integrations.gs.common import (
    Constants,
    GCSStepLock,
    get_client,
    get_credentials,
)
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

    def __init__(self, workspace: str, project: Optional[str] = None, **kwargs):
        credentials = get_credentials()
        print("Credentials", credentials)
        print("Project", project, credentials.quota_project_id)
        self._client = get_client(gcs_workspace=workspace, project=project, **kwargs)
        self._cache = GSStepCache(workspace, client=self._client)
        self._locks: Dict[Step, GCSStepLock] = {}

        self._step_info_cache: "OrderedDict[str, StepInfo]" = OrderedDict()
        super().__init__()

        print("Storage client")
        print("Credentials", self._client.storage._credentials)
        print("Project", self._client.storage._credentials.quota_project_id)

        # TODO: Ugly. Fix.
        # TODO: also update the docstring.
        credentials = get_credentials()
        print("Credentials", credentials)
        print("Project", project, credentials.quota_project_id)
        self._ds = datastore.Client(namespace=workspace, project=project, credentials=credentials)

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

    def step_info(self, step_or_unique_id: Union[Step, str]) -> StepInfo:
        unique_id = (
            step_or_unique_id if isinstance(step_or_unique_id, str) else step_or_unique_id.unique_id
        )
        step_info_entity = self._ds.get(key=self._ds.key("stepinfo", unique_id))
        if step_info_entity:
            # TODO: not using self._step_info_cache yet.
            # TODO: why does it use digest rather than unique id?
            step_info_bytes = step_info_entity["step_info_dict"]
            step_info = StepInfo.from_json_dict(json.loads(step_info_bytes))
            return step_info
        else:
            if not isinstance(step_or_unique_id, Step):
                raise KeyError(step_or_unique_id)
            step_info = StepInfo.new_from_step(step_or_unique_id)
            self._update_step_info(step_info)
            return step_info

    def _update_step_info(self, step_info: StepInfo):

        step_info_entity = self._ds.entity(
            key=self._ds.key("stepinfo", step_info.unique_id),
            exclude_from_indexes=("step_info_dict",),
        )

        # We can store each key separately, but we only index things that are useful for querying.
        # TODO: do we want to index any other step_info keys?

        step_info_entity["step_name"] = step_info.step_name
        step_info_entity["start_time"] = step_info.start_time
        step_info_entity["end_time"] = step_info.end_time
        step_info_entity["result_location"] = step_info.result_location
        step_info_entity["step_info_dict"] = json.dumps(step_info.to_json_dict()).encode()

        self._ds.put(step_info_entity)

    @contextmanager
    def capture_logs_for_run(self, name: str) -> Generator[None, None, None]:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            log_file = Path(tmp_dir_name) / "out.log"
            try:
                with file_handler(log_file):
                    yield None
            finally:
                run_dataset = self.Constants.run_dataset_name(name)
                self.client.sync(run_dataset, log_file)
                # TODO: temp for testing
                # Not committing since Run datasets now different.
                # self.client.commit(run_dataset)
