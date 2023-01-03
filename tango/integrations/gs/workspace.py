from collections import OrderedDict
from typing import Dict, TypeVar
from urllib.parse import ParseResult

from tango.step import Step
from tango.step_info import StepInfo
from tango.workspace import Workspace
from tango.workspaces.remote_workspace import RemoteWorkspace

from .common import Constants, GCSStepLock, get_client
from .step_cache import GSStepCache

T = TypeVar("T")


@Workspace.register("gs")
class GSWorkspace(RemoteWorkspace):
    Constants = Constants
    """
    Assumes that you have run `gcloud auth application-default login`
    """

    def __init__(self, workspace: str, **kwargs):
        self._client = get_client(gcs_workspace=workspace, **kwargs)
        self._cache = GSStepCache(workspace, client=self._client)
        self._locks: Dict[Step, GCSStepLock] = {}

        self._step_info_cache: "OrderedDict[str, StepInfo]" = OrderedDict()
        super().__init__()

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
            # e.g. "beaker://ai2/my-workspace"
            workspace = parsed_url.netloc + parsed_url.path
        elif parsed_url.netloc:
            # e.g. "beaker://my-workspace"
            workspace = parsed_url.netloc
        else:
            raise ValueError(f"Bad URL for GCS workspace '{parsed_url}'")
        return cls(workspace)

    @property
    def url(self) -> str:
        return f"gs://{self.client.full_name}"

    def _remote_lock(self, step: Step) -> GCSStepLock:
        return GCSStepLock(self.client, step)

    def _dataset_url(self, workspace_url: str, dataset_name: str) -> str:
        return self.client.dataset_url(workspace_url, dataset_name)
