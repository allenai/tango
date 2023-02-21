import datetime
import json
import random
from pathlib import Path
from typing import Dict, Optional, TypeVar, Union, cast
from urllib.parse import ParseResult

import petname
from google.cloud import datastore
from google.oauth2.credentials import Credentials

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

    :param workspace: The name or ID of the Google Cloud bucket to use.
    :param project: The Google project ID. This is required for the datastore. If not provided,
        it will be inferred from the Google cloud credentials.

    .. important::
        Credentials can be provided in the following ways:

        - Using the `credentials` keyword argument:
            - You can specify the path to the credentials json file.
            - You can specify the `google.oauth2.credentials.Credentials()` object.
            - You can specify the json string of credentials dict.

        - Using the default credentials: You can use your default google cloud credentials by running
          `gcloud auth application-default login`. If you are using `GSWorkspace` with
          :class:`~tango.integrations.beaker.BeakerExecutor`, you will need to set the environment variable
          `GOOGLE_TOKEN` to the credentials json file. The default location is usually
          `~/.config/gcloud/application_default_credentials.json`.

    """

    Constants = Constants
    NUM_CONCURRENT_WORKERS = 32

    def __init__(
        self,
        workspace: str,
        project: Optional[str] = None,
        credentials: Optional[Union[str, Credentials]] = None,
    ):
        self.client = get_client(bucket_name=workspace, credentials=credentials, project=project)

        self.client.NUM_CONCURRENT_WORKERS = self.NUM_CONCURRENT_WORKERS
        self._cache = GSStepCache(workspace, client=self.client)
        self._locks: Dict[Step, GCSStepLock] = {}

        super().__init__()

        credentials = get_credentials()
        project = project or credentials.quota_project_id
        self._ds = datastore.Client(namespace=workspace, project=project, credentials=credentials)

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

    def _step_location(self, step: Step) -> str:
        return self.client.url(self.Constants.step_artifact_name(step))

    def _save_run(
        self, steps: Dict[str, StepInfo], run_data: Dict[str, str], name: Optional[str] = None
    ) -> Run:
        if name is None:
            while True:
                name = petname.generate() + str(random.randint(0, 100))
                if not self._ds.get(self._ds.key("run", name)):
                    break
        else:
            if self._ds.get(self._ds.key("run", name)):
                raise ValueError(f"Run name '{name}' is already in use")

        run_entity = self._ds.entity(key=self._ds.key("run", name), exclude_from_indexes=("steps",))
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
        step_info_entity["step_name"] = step_info.step_name
        step_info_entity["start_time"] = step_info.start_time
        step_info_entity["end_time"] = step_info.end_time
        step_info_entity["result_location"] = step_info.result_location
        step_info_entity["step_info_dict"] = json.dumps(step_info.to_json_dict()).encode()

        self._ds.put(step_info_entity)

    def _save_run_log(self, name: str, log_file: Path):
        """
        The logs are stored in the bucket. The Run object details are stored in
        the remote database.
        """
        run_dataset = self.Constants.run_artifact_name(name)
        self.client.upload(run_dataset, log_file)
