from typing import Union

from tango.step import Step
from tango.step_caches.remote_step_cache import RemoteConstants
from tango.step_info import StepInfo

from .util import GCSClient


def get_client(gcs_workspace: str, **kwargs) -> GCSClient:
    return GCSClient(gcs_workspace, **kwargs)


# TODO: duplicate code. move some place else.


class Constants(RemoteConstants):
    pass


def step_dataset_name(step: Union[str, StepInfo, Step]) -> str:
    return f"{Constants.STEP_DATASET_PREFIX}{step if isinstance(step, str) else step.unique_id}"


def step_lock_dataset_name(step: Union[str, StepInfo, Step]) -> str:
    return f"{step_dataset_name(step)}-lock"


def run_dataset_name(name: str) -> str:
    return f"{Constants.RUN_DATASET_PREFIX}{name}"
