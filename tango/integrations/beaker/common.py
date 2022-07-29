import logging
import time
import urllib.parse
from typing import Optional, Union

from beaker import Beaker, Dataset, DatasetConflict, DatasetNotFound

from tango.step import Step
from tango.step_info import StepInfo

logger = logging.getLogger(__name__)


class Constants:
    RUN_DATASET_PREFIX = "tango-run-"
    RUN_DATA_FNAME = "run.json"
    STEP_DATASET_PREFIX = "tango-step-"
    STEP_INFO_FNAME = "step_info.json"
    STEP_RESULT_DIR = "result"
    ENTRYPOINT_DATASET_PREFIX = "tango-entrypoint-"
    STEP_GRAPH_DATASET_PREFIX = "tango-step-graph-"
    STEP_EXPERIMENT_PREFIX = "tango-step-"


def step_dataset_name(step: Union[str, StepInfo, Step]) -> str:
    return f"{Constants.STEP_DATASET_PREFIX}{step if isinstance(step, str) else step.unique_id}"


def step_lock_dataset_name(step: Union[str, StepInfo, Step]) -> str:
    return f"{step_dataset_name(step)}-lock"


def run_dataset_name(name: str) -> str:
    return f"{Constants.RUN_DATASET_PREFIX}{name}"


def dataset_url(workspace_url: str, dataset_name: str) -> str:
    return (
        workspace_url
        + "/datasets?"
        + urllib.parse.urlencode(
            {
                "text": dataset_name,
                "committed": "false",
            }
        )
    )


class BeakerStepLock:
    def __init__(self, beaker: Beaker, step: Union[str, StepInfo, Step], **kwargs):
        self._beaker = beaker
        self._step_id = step if isinstance(step, str) else step.unique_id
        self._lock_dataset_name = step_lock_dataset_name(step)
        self._lock_dataset: Optional[Dataset] = None
        self.lock_dataset_url = dataset_url(beaker.workspace.url(), self._lock_dataset_name)

    def acquire(self, timeout=None, poll_interval: float = 2.0, log_interval: float = 30.0) -> None:
        if self._lock_dataset is not None:
            return
        start = time.monotonic()
        last_logged = None
        while timeout is None or (time.monotonic() - start < timeout):
            try:
                self._lock_dataset = self._beaker.dataset.create(
                    self._lock_dataset_name, commit=False
                )
            except DatasetConflict:
                if last_logged is None or last_logged - start >= log_interval:
                    logger.warning(
                        "Waiting to acquire lock dataset for step '%s':\n\n%s\n\n"
                        "This probably means the step is being run elsewhere, but if you're sure it isn't "
                        "you can just delete the lock dataset.",
                        self._step_id,
                        self.lock_dataset_url,
                    )
                    last_logged = time.monotonic()
                time.sleep(poll_interval)
                continue
            else:
                break
        else:
            raise TimeoutError(
                f"Timeout error occurred while waiting to acquire dataset lock for step '{self._step_id}':\n\n"
                f"{self.lock_dataset_url}\n\n"
                f"This probably means the step is being run elsewhere, but if you're sure it isn't you can "
                f"just delete the lock dataset."
            )

    def release(self):
        if self._lock_dataset is not None:
            try:
                self._beaker.dataset.delete(self._lock_dataset)
            except DatasetNotFound:
                # Dataset must have been manually deleted.
                pass
            self._lock_dataset = None

    def __del__(self):
        self.release()
