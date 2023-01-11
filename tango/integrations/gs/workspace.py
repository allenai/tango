from collections import OrderedDict
from typing import Dict, TypeVar
from urllib.parse import ParseResult

from tango.integrations.gs.common import Constants, GCSStepLock, get_client
from tango.integrations.gs.step_cache import GSStepCache
from tango.step import Step
from tango.step_info import StepInfo
from tango.workspace import Workspace
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
