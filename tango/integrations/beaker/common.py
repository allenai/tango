from typing import Union

from tango.step import Step
from tango.step_info import StepInfo


class Constants:
    RUN_DATASET_PREFIX = "tango-run-"
    RUN_DATA_FNAME = "run.json"
    STEP_DATASET_PREFIX = "tango-step-"
    STEP_INFO_FNAME = "step_info.json"
    STEP_RESULT_DIR = "result"


def step_dataset_name(step: Union[str, StepInfo, Step]) -> str:
    return f"{Constants.STEP_DATASET_PREFIX}{step if isinstance(step, str) else step.unique_id}"


def run_dataset_name(name: str) -> str:
    return f"{Constants.RUN_DATASET_PREFIX}{name}"
