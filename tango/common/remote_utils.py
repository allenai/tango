import logging
from typing import Union

from tango.step import Step
from tango.step_info import StepInfo

logger = logging.getLogger(__name__)


class RemoteConstants:
    """
    Common constants to be used as prefixes and filenames in remote workspaces.
    """

    RUN_ARTIFACT_PREFIX = "tango-run-"
    RUN_DATA_FNAME = "run.json"
    STEP_ARTIFACT_PREFIX = "tango-step-"
    STEP_INFO_FNAME = "step_info.json"
    STEP_RESULT_DIR = "result"
    STEP_GRAPH_ARTIFACT_PREFIX = "tango-step-graph-"
    STEP_EXPERIMENT_PREFIX = "tango-step-"
    STEP_GRAPH_FILENAME = "config.json"
    GITHUB_TOKEN_SECRET_NAME: str = "TANGO_GITHUB_TOKEN"
    RESULTS_DIR: str = "/tango/output"
    INPUT_DIR: str = "/tango/input"
    LOCK_ARTIFACT_SUFFIX: str = "-lock"

    @classmethod
    def step_artifact_name(cls, step: Union[str, StepInfo, Step]) -> str:
        return f"{cls.STEP_ARTIFACT_PREFIX}{step if isinstance(step, str) else step.unique_id}"

    @classmethod
    def step_lock_artifact_name(cls, step: Union[str, StepInfo, Step]) -> str:
        return f"{cls.step_artifact_name(step)}{cls.LOCK_ARTIFACT_SUFFIX}"

    @classmethod
    def run_artifact_name(cls, name: str) -> str:
        return f"{cls.RUN_ARTIFACT_PREFIX}{name}"
