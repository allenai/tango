import json
import random
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Optional, TypeVar, Union, cast
from urllib.parse import ParseResult

import petname
from google.auth.credentials import Credentials
from google.cloud import datastore

from tango.common.util import utc_now_datetime
from tango.integrations.gs.common import (
    Constants,
    GCSStepLock,
    get_bucket_and_prefix,
    get_client,
    get_credentials,
)
from tango.integrations.gs.step_cache import GSStepCache
from tango.step import Step
from tango.step_info import StepInfo, StepState
from tango.workspace import Run, RunInfo, RunSort, StepInfoSort, Workspace
from tango.workspaces.remote_workspace import RemoteWorkspace

T = TypeVar("T")


@Workspace.register("gs")
class GSWorkspace(RemoteWorkspace):
    """
    This is a :class:`~tango.workspace.Workspace` that stores step artifacts on Google Cloud Storage.

    .. tip::
        Registered as a :class:`~tango.workspace.Workspace` under the name "gs".

    :param workspace: The name or ID of the Google Cloud bucket folder to use.
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
        self.client = get_client(folder_name=workspace, credentials=credentials, project=project)

        self.client.NUM_CONCURRENT_WORKERS = self.NUM_CONCURRENT_WORKERS
        self._cache = GSStepCache(workspace, client=self.client)
        self._locks: Dict[Step, GCSStepLock] = {}

        super().__init__()

        credentials = get_credentials()
        project = project or credentials.quota_project_id

        self.bucket_name, self.prefix = get_bucket_and_prefix(workspace)
        self._ds = datastore.Client(
            namespace=self.bucket_name, project=project, credentials=credentials
        )

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

    @property
    def _run_key(self):
        return self.client._gs_path("run")

    @property
    def _stepinfo_key(self):
        return self.client._gs_path("stepinfo")

    def _save_run(
        self, steps: Dict[str, StepInfo], run_data: Dict[str, str], name: Optional[str] = None
    ) -> Run:
        if name is None:
            while True:
                name = petname.generate() + str(random.randint(0, 100))
                if not self._ds.get(self._ds.key(self._run_key, name)):
                    break
        else:
            if self._ds.get(self._ds.key(self._run_key, name)):
                raise ValueError(f"Run name '{name}' is already in use")

        run_entity = self._ds.entity(
            key=self._ds.key(self._run_key, name), exclude_from_indexes=("steps",)
        )
        # Even though the run's name is part of the key, we add this as a
        # field so we can index on it and order asc/desc (indices on the key field don't allow ordering).
        run_entity["name"] = name
        run_entity["start_date"] = utc_now_datetime()
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
            for run_entity in self._ds.query(kind=self._run_key).fetch():
                run_futures.append(executor.submit(self._get_run_from_entity, run_entity))
            for future in concurrent.futures.as_completed(run_futures):
                run = future.result()
                if run is not None:
                    runs[run.name] = run

        return runs

    def search_registered_runs(
        self,
        *,
        sort_by: Optional[RunSort] = None,
        sort_descending: bool = True,
        match: Optional[str] = None,
        start: int = 0,
        stop: Optional[int] = None,
    ) -> List[RunInfo]:
        run_entities = self._fetch_run_entities(
            sort_by=sort_by, sort_descending=sort_descending, match=match, start=start, stop=stop
        )
        return [
            RunInfo(name=e.key.name, start_date=e["start_date"], steps=json.loads(e["steps"]))
            for e in run_entities
        ]

    def num_registered_runs(self, *, match: Optional[str] = None) -> int:
        count = 0
        for _ in self._fetch_run_entities(match=match):
            count += 1
        return count

    def _fetch_run_entities(
        self,
        *,
        sort_by: Optional[RunSort] = None,
        sort_descending: bool = True,
        match: Optional[str] = None,
        start: int = 0,
        stop: Optional[int] = None,
    ) -> Generator[datastore.Entity, None, None]:
        from itertools import islice

        # Note: we can't query or order by multiple fields without a suitable
        # composite index. So in that case we have to apply remaining filters
        # or slice and order locally. We'll default to using 'match' in the query.
        # But if 'match' is null we can sort with the query.
        sort_locally = bool(match)

        sort_field: Optional[str] = None
        if sort_by == RunSort.START_DATE:
            sort_field = "start_date"
        elif sort_by == RunSort.NAME:
            sort_field = "name"
        elif sort_by is not None:
            raise NotImplementedError(sort_by)

        order: List[str] = []
        if sort_field is not None and not sort_locally:
            order = [sort_field if not sort_descending else f"-{sort_field}"]

        query = self._ds.query(kind=self._run_key, order=order)
        if match:
            # HACK: Datastore has no direct string matching functionality,
            # but this comparison is equivalent to checking if 'name' starts with 'match'.
            query.add_filter("name", ">=", match)
            query.add_filter("name", "<=", match[:-1] + chr(ord(match[-1]) + 1))

        entity_iter: Iterable[datastore.Entity] = query.fetch(
            offset=0 if sort_locally else start,
            limit=None if (stop is None or sort_locally) else stop - start,
        )

        if sort_field is not None and sort_locally:
            entity_iter = sorted(
                entity_iter, key=lambda entity: entity[sort_field], reverse=sort_descending
            )

        if sort_locally:
            entity_iter = islice(entity_iter, start, stop)

        for entity in entity_iter:
            yield entity

    def search_step_info(
        self,
        *,
        sort_by: Optional[StepInfoSort] = None,
        sort_descending: bool = True,
        match: Optional[str] = None,
        state: Optional[StepState] = None,
        start: int = 0,
        stop: Optional[int] = None,
    ) -> List[StepInfo]:
        step_info_entities = self._fetch_step_info_entities(
            sort_by=sort_by,
            sort_descending=sort_descending,
            match=match,
            state=state,
            start=start,
            stop=stop,
        )
        return [
            StepInfo.from_json_dict(json.loads(e["step_info_dict"])) for e in step_info_entities
        ]

    def num_steps(self, *, match: Optional[str] = None, state: Optional[StepState] = None) -> int:
        count = 0
        for _ in self._fetch_step_info_entities(match=match, state=state):
            count += 1
        return count

    def _fetch_step_info_entities(
        self,
        *,
        sort_by: Optional[StepInfoSort] = None,
        sort_descending: bool = True,
        match: Optional[str] = None,
        state: Optional[StepState] = None,
        start: int = 0,
        stop: Optional[int] = None,
    ) -> Generator[datastore.Entity, None, None]:
        from itertools import islice

        # Note: we can't query or order by multiple fields without a suitable
        # composite index. So in that case we have to apply remaining filters
        # or slice and order locally. We'll default to using 'match' in the query.
        # But if 'match' is null, we'll use 'state' to filter in the query.
        # If 'state' is also null, we can sort with the query.
        sort_locally = sort_by is not None and (match is not None or state is not None)
        filter_locally = state is not None and match is not None
        slice_locally = sort_locally or filter_locally

        sort_field: Optional[str] = None
        if sort_by == StepInfoSort.START_TIME:
            sort_field = "start_time"
        elif sort_by == StepInfoSort.UNIQUE_ID:
            sort_field = "step_id"
        elif sort_by is not None:
            raise NotImplementedError(sort_by)

        order: List[str] = []
        if sort_field is not None and not sort_locally:
            order = [sort_field if not sort_descending else f"-{sort_field}"]

        query = self._ds.query(kind=self._stepinfo_key, order=order)

        if match is not None:
            # HACK: Datastore has no direct string matching functionality,
            # but this comparison is equivalent to checking if 'step_id' starts with 'match'.
            query.add_filter("step_id", ">=", match)
            query.add_filter("step_id", "<=", match[:-1] + chr(ord(match[-1]) + 1))
        elif state is not None and not filter_locally:
            query.add_filter("state", "=", str(state.value))

        entity_iter: Iterable[datastore.Entity] = query.fetch(
            offset=0 if slice_locally else start,
            limit=None if (stop is None or slice_locally) else stop - start,
        )

        if state is not None and filter_locally:
            entity_iter = filter(lambda entity: entity["state"] == state, entity_iter)

        if sort_field is not None and sort_locally:
            entity_iter = sorted(
                entity_iter, key=lambda entity: entity[sort_field], reverse=sort_descending
            )

        if slice_locally:
            entity_iter = islice(entity_iter, start, stop)

        for entity in entity_iter:
            yield entity

    def registered_run(self, name: str) -> Run:
        err_msg = f"Run '{name}' not found in workspace"

        run_entity = self._ds.get(key=self._ds.key(self._run_key, name))
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
        step_info_entity = self._ds.get(key=self._ds.key(self._stepinfo_key, unique_id))
        if step_info_entity is not None:
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
            key=self._ds.key(self._stepinfo_key, step_info.unique_id),
            exclude_from_indexes=("step_info_dict",),
        )

        # Even though the step's unique ID is part of the key, we add this as a
        # field so we can index on it and order asc/desc (indices on the key field don't allow ordering).
        step_info_entity["step_id"] = step_info.unique_id
        step_info_entity["step_name"] = step_info.step_name
        step_info_entity["start_time"] = step_info.start_time
        step_info_entity["end_time"] = step_info.end_time
        step_info_entity["state"] = str(step_info.state.value)
        step_info_entity["updated"] = utc_now_datetime()
        step_info_entity["step_info_dict"] = json.dumps(step_info.to_json_dict()).encode()

        self._ds.put(step_info_entity)

    def _remove_step_info(self, step_info: StepInfo) -> None:
        # remove dir from bucket
        step_artifact = self.client.get(self.Constants.step_artifact_name(step_info))
        if step_artifact is not None:
            self.client.delete(step_artifact)

        # remove datastore entities
        self._ds.delete(key=self._ds.key("stepinfo", step_info.unique_id))

    def _save_run_log(self, name: str, log_file: Path):
        """
        The logs are stored in the bucket. The Run object details are stored in
        the remote database.
        """
        run_dataset = self.Constants.run_artifact_name(name)
        self.client.upload(run_dataset, log_file)
