import atexit
import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

from tango.step import Step
from tango.step_caches.remote_step_cache import RemoteConstants
from tango.step_info import StepInfo

from .util import GCSClient, GCSDataset, GCSDatasetConflict, GCSDatasetNotFound

logger = logging.getLogger(__name__)


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


def dataset_url(workspace_url: str, dataset_name: str) -> str:
    return workspace_url + "/ " + dataset_name


class GCSStepLock:
    METADATA_FNAME = "metadata.json"

    def __init__(
        self,
        client: GCSClient,
        step: Union[str, StepInfo, Step],
    ):
        self._client = client
        self._step_id = step if isinstance(step, str) else step.unique_id
        self._lock_dataset_name = step_lock_dataset_name(step)
        self._lock_dataset: Optional[GCSDataset] = None
        self.lock_dataset_url = dataset_url(client.url(), self._lock_dataset_name)

    @property
    def metadata(self) -> Dict[str, Any]:
        # TODO: should we put something here?
        return {}

    def acquire(self, timeout=None, poll_interval: float = 2.0, log_interval: float = 30.0) -> None:
        if self._lock_dataset is not None:
            return
        start = time.monotonic()
        last_logged = None
        while timeout is None or (time.monotonic() - start < timeout):
            try:
                self._lock_dataset = self._client.create(self._lock_dataset_name, commit=False)

                atexit.register(self.release)

                # Write metadata.
                with tempfile.TemporaryDirectory() as tmp_dir_name:
                    tmp_dir = Path(tmp_dir_name)
                    metadata_path = tmp_dir / self.METADATA_FNAME
                    with open(metadata_path, "w") as f:
                        json.dump(self.metadata, f)
                    self._client.sync(self._lock_dataset, metadata_path)  # type: ignore
            except GCSDatasetConflict:

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
                self._client.delete(self._lock_dataset)
            except GCSDatasetNotFound:
                # Dataset must have been manually deleted.
                pass
            self._lock_dataset = None
            atexit.unregister(self.release)

    def __del__(self):
        self.release()
