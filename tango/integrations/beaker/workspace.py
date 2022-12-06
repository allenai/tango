import os
from collections import OrderedDict
from typing import Dict, Optional, TypeVar
from urllib.parse import ParseResult

from beaker import Experiment, ExperimentNotFound

from tango.step import Step
from tango.step_info import StepInfo
from tango.workspace import Workspace
from tango.workspaces.remote_workspace import RemoteWorkspace

from .common import BeakerStepLock, dataset_url, get_client
from .step_cache import BeakerStepCache

T = TypeVar("T")


@Workspace.register("beaker")
class BeakerWorkspace(RemoteWorkspace):
    """
    This is a :class:`~tango.workspace.Workspace` that stores step artifacts on `Beaker`_.

    .. tip::
        Registered as a :class:`~tango.workspace.Workspace` under the name "beaker".

    :param beaker_workspace: The name or ID of the Beaker workspace to use.
    :param kwargs: Additional keyword arguments passed to :meth:`Beaker.from_env() <beaker.Beaker.from_env()>`.
    """

    STEP_INFO_CACHE_SIZE = 512

    def __init__(self, workspace: str, **kwargs):
        client = get_client(workspace, **kwargs)
        # TODO: for the time being
        cache = BeakerStepCache(workspace, beaker=client.beaker)
        locks: Dict[Step, BeakerStepLock] = {}
        self._step_info_cache: "OrderedDict[str, StepInfo]" = OrderedDict()
        super().__init__(client, cache, "beaker_workspace", locks)

    @property
    def beaker(self):
        # TODO: for the time being.
        return self.client.beaker

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

    def _remote_lock(self, step: Step) -> BeakerStepLock:  # type: ignore
        # TODO: deal with mypy remotesteplock
        return BeakerStepLock(
            self.beaker, step, current_beaker_experiment=self.current_beaker_experiment
        )

    @classmethod
    def _dataset_url(cls, workspace_url: str, dataset_name: str) -> str:
        return dataset_url(workspace_url, dataset_name)
