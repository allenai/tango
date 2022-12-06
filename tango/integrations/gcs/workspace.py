from collections import OrderedDict
from typing import Dict, TypeVar

from tango.step import Step
from tango.step_info import StepInfo
from tango.workspace import Workspace
from tango.workspaces.remote_workspace import RemoteWorkspace

from .common import Constants, GCSStepLock, get_client
from .step_cache import GCSStepCache

T = TypeVar("T")


@Workspace.register("gs")
class GCSWorkspace(RemoteWorkspace):
    Constants = Constants
    """
    Assumes that you have run `gcloud auth application-default login`
    """

    def __init__(self, workspace: str, **kwargs):
        client = get_client(gcs_workspace=workspace, **kwargs)
        cache = GCSStepCache(workspace, client=client)
        locks: Dict[Step, GCSStepLock] = {}
        self._step_info_cache: "OrderedDict[str, StepInfo]" = OrderedDict()
        super().__init__(client, cache, "gs_workspace", locks)

    @property
    def url(self) -> str:
        return f"gs://{self.client.full_name}"

    def _remote_lock(self, step: Step) -> GCSStepLock:
        return GCSStepLock(self.client, step)

    def _dataset_url(self, workspace_url: str, dataset_name: str) -> str:
        return self.client.dataset_url(workspace_url, dataset_name)
