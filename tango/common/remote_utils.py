import atexit
import datetime
import json
import logging
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

from tango.common.exceptions import TangoError
from tango.step import Step
from tango.step_info import StepInfo

logger = logging.getLogger(__name__)


class RemoteConstants:
    """
    Common constants to be used as prefixes and filenames in remote workspaces.
    """

    RUN_DATASET_PREFIX = "tango-run-"
    RUN_DATA_FNAME = "run.json"
    STEP_DATASET_PREFIX = "tango-step-"
    STEP_INFO_FNAME = "step_info.json"
    STEP_RESULT_DIR = "result"
    STEP_GRAPH_DATASET_PREFIX = "tango-step-graph-"
    STEP_EXPERIMENT_PREFIX = "tango-step-"
    STEP_GRAPH_FILENAME = "config.json"
    GITHUB_TOKEN_SECRET_NAME: str = "TANGO_GITHUB_TOKEN"
    RESULTS_DIR: str = "/tango/output"
    INPUT_DIR: str = "/tango/input"

    @classmethod
    def step_dataset_name(cls, step: Union[str, StepInfo, Step]) -> str:
        return f"{cls.STEP_DATASET_PREFIX}{step if isinstance(step, str) else step.unique_id}"

    @classmethod
    def step_lock_dataset_name(cls, step: Union[str, StepInfo, Step]) -> str:
        return f"{cls.step_dataset_name(step)}-lock"

    @classmethod
    def run_dataset_name(cls, name: str) -> str:
        return f"{cls.RUN_DATASET_PREFIX}{name}"


@dataclass
class RemoteDataset:

    name: str
    dataset_path: str
    created: datetime.datetime
    committed: bool


@dataclass
class RemoteFileInfo:
    # TODO: this is just mirroring beaker right now. We may not need this level of abstraction.
    path: str
    digest: str
    updated: datetime.datetime
    size: int


class RemoteDatasetConflict(TangoError):
    pass


class RemoteDatasetNotFound(TangoError):
    pass


class RemoteClient:
    """
    A client for interacting with remote storage.
    """

    def __init__(self, *args, **kwargs):
        pass

    def url(self, dataset: Optional[str] = None):
        raise NotImplementedError()

    def dataset_url(self, workspace_url: str, dataset_name: str) -> str:
        raise NotImplementedError()

    @property
    def full_name(self):
        raise NotImplementedError()

    def get(self, dataset) -> RemoteDataset:
        raise NotImplementedError()

    def create(self, dataset: str, commit: bool = False):
        raise NotImplementedError()

    def delete(self, dataset):
        raise NotImplementedError()

    def sync(self, dataset, objects_dir):
        raise NotImplementedError()

    def commit(self, dataset):
        raise NotImplementedError()

    def upload(self, dataset, source, target) -> None:
        raise NotImplementedError()

    def get_file(self, dataset, file_path):
        raise NotImplementedError()

    def file_info(self, dataset, file_path) -> RemoteFileInfo:
        raise NotImplementedError()

    def fetch(self, dataset, target_dir) -> RemoteDataset:
        raise NotImplementedError()

    def datasets(self, match: str, uncommitted: bool = False, results: bool = False):
        raise NotImplementedError()


class RemoteStepLock:
    METADATA_FNAME = "metadata.json"

    def __init__(
        self,
        client: RemoteClient,
        step: Union[str, StepInfo, Step],
    ):
        self._client = client
        self._step_id = step if isinstance(step, str) else step.unique_id
        self._lock_dataset_name = RemoteConstants.step_lock_dataset_name(step)
        self._lock_dataset: Optional[RemoteDataset] = None
        self.lock_dataset_url = self._dataset_url(client.url(), self._lock_dataset_name)

    @classmethod
    def _dataset_url(cls, workspace_url: str, lock_dataset_name: str) -> str:
        # TODO: this should be the client's method, change when Beaker is also a RemoteClient.
        raise NotImplementedError()

    @property
    def metadata(self) -> Dict[str, Any]:
        """
        Currently, this is used for the Beaker implementation mostly, to record Beaker's experiment name.
        In the future, we can add other relevant metadata.
        """
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
            except RemoteDatasetConflict:

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
            except RemoteDatasetNotFound:
                # Dataset must have been manually deleted.
                pass
            self._lock_dataset = None
            atexit.unregister(self.release)

    def __del__(self):
        self.release()
