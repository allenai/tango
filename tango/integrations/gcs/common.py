import logging
from typing import Union

from tango.common.remote_utils import RemoteConstants, RemoteStepLock
from tango.step import Step
from tango.step_info import StepInfo

from .util import GCSClient

logger = logging.getLogger(__name__)


def get_client(gcs_workspace: str, **kwargs) -> GCSClient:
    return GCSClient(gcs_workspace, **kwargs)


class Constants(RemoteConstants):
    pass


class GCSStepLock(RemoteStepLock):
    def __init__(self, client, step: Union[str, StepInfo, Step]):
        super().__init__(client, step)

    @classmethod
    def _dataset_url(cls, workspace_url: str, lock_dataset_name: str) -> str:
        return workspace_url + "/ " + lock_dataset_name
