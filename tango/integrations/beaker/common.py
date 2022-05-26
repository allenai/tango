import time
from typing import Optional, Union

from beaker import Beaker, Dataset, DatasetConflict
from filelock import AcquireReturnProxy

from tango.common.aliases import PathOrStr
from tango.common.file_lock import FileLock
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


def step_lock_dataset_name(step: Union[str, StepInfo, Step]) -> str:
    return f"{step_dataset_name(step)}-lock"


def run_dataset_name(name: str) -> str:
    return f"{Constants.RUN_DATASET_PREFIX}{name}"


class BeakerLock(FileLock):
    def __init__(self, lock_file: PathOrStr, beaker: Beaker, lock_dataset: str, **kwargs):
        super().__init__(lock_file, **kwargs)
        self._beaker = beaker
        self._lock_dataset_name = lock_dataset
        self._lock_dataset: Optional[Dataset] = None

    def acquire(self, timeout=None, poll_interval=0.05) -> AcquireReturnProxy:
        start = time.monotonic()
        while timeout is None or (time.monotonic() - start < timeout):
            try:
                self._lock_dataset = self._beaker.dataset.create(self._lock_dataset_name)
            except DatasetConflict:
                continue
            else:
                break
        else:
            raise TimeoutError(
                f"Timeout error waiting to acquire dataset lock "
                f"{self._beaker.dataset.url(self.lock_dataset_name)}"
            )
        return super().acquire(timeout=timeout, poll_interval=poll_interval)

    def release(self, force: bool = False):
        if self._lock_dataset is not None:
            self._beaker.dataset.delete(self._lock_dataset)
            self._lock_dataset = None
        super().release(force=force)
