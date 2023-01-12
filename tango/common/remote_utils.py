import atexit
import datetime
import json
import logging
import tempfile
import time
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from tango.common import PathOrStr
from tango.common.exceptions import TangoError
from tango.step import Step
from tango.step_info import StepInfo

from .registrable import Registrable

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
    LOCK_DATASET_SUFFIX: str = "-lock"

    @classmethod
    def step_dataset_name(cls, step: Union[str, StepInfo, Step]) -> str:
        return f"{cls.STEP_DATASET_PREFIX}{step if isinstance(step, str) else step.unique_id}"

    @classmethod
    def step_lock_dataset_name(cls, step: Union[str, StepInfo, Step]) -> str:
        return f"{cls.step_dataset_name(step)}{cls.LOCK_DATASET_SUFFIX}"

    @classmethod
    def run_dataset_name(cls, name: str) -> str:
        return f"{cls.RUN_DATASET_PREFIX}{name}"


@dataclass
class RemoteDataset:
    """
    Abstraction for all objects in remote workspaces. Conceptually, this can be thought of as a folder.
    """

    name: str
    """
    Name of the dataset.
    """
    dataset_path: str
    """
    Remote location url for the dataset.
    """
    created: datetime.datetime
    """
    Time of creation.
    """


@dataclass
class RemoteFileInfo:
    """
    Abstraction for file objects in remote workspaces.

    Note: this is just mirroring beaker right now. We may not need this level of abstraction.
    """

    path: str
    """
    Remote location url for the file.
    """
    digest: str
    """
    Hash string representing the file.
    """
    updated: datetime.datetime
    """
    Last update time of the file.
    """
    size: int
    """
    Size of the file.
    """


class RemoteDatasetConflict(TangoError):
    """
    Error denoting that the remote dataset already exists.
    """

    pass


class RemoteDatasetNotFound(TangoError):
    """
    Error denoting that the remote dataset does not exist.
    """

    pass


class RemoteDatasetWriteError(TangoError):
    """
    Error denoting that there was an issue writing the dataset to its remote location.
    """

    pass


class RemoteClient(Registrable):
    """
    A client for interacting with remote storage. All remote clients inherit from this.
    """

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def url(self, dataset: Optional[str] = None) -> str:
        """
        Returns the remote url of the `dataset`.
        """
        raise NotImplementedError()

    @abstractmethod
    def get(self, dataset) -> RemoteDataset:
        """
        Returns a `RemoteDataset` object created by fetching the dataset's information
        from remote location.
        """
        raise NotImplementedError()

    @abstractmethod
    def create(self, dataset: str):
        """
        Creates a new dataset in the remote location.
        """
        raise NotImplementedError()

    @abstractmethod
    def delete(self, dataset):
        """
        Removes a dataset from the remote location.
        """
        raise NotImplementedError()

    @abstractmethod
    def sync(self, dataset, objects_dir):
        """
        Writes the contents of objects_dir to the remote dataset location.
        """
        raise NotImplementedError()

    @abstractmethod
    def upload(self, dataset, source: bytes, target: PathOrStr) -> None:
        """
        Uploads the `source` contents to the `target` file within the remote dataset.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_file(self, dataset, file_path):
        """
        Returns the file contents at the `file_path` within the remote dataset.
        """
        raise NotImplementedError()

    @abstractmethod
    def file_info(self, dataset, file_path) -> RemoteFileInfo:
        """
        Returns a `RemoteFileInfo` object constructed from `file_path` within the remote dataset location.
        """
        raise NotImplementedError()

    @abstractmethod
    def fetch(self, dataset, target_dir: PathOrStr) -> RemoteDataset:
        """
        Writes the contents of the remote dataset to the `target_dir`.
        """
        raise NotImplementedError()

    @abstractmethod
    def list_steps(self, match: str) -> List:
        """
        Lists the steps within the workspace attached to the client, based on `match`
        criteria.
        """
        raise NotImplementedError()

    @abstractmethod
    def list_runs(self, match: str) -> List:
        """
        Lists the runs within the workspace attached to the client, based on `match`
        criteria.
        """
        raise NotImplementedError()


class RemoteStepLock:
    """
    Utility class for handling locking mechanism for :class:`~tango.step.Step` object stored at a
    remote location as a `RemoteDataset`.
    """

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
        self.lock_dataset_url = self._client.url(self._lock_dataset_name)

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
                self._lock_dataset = self._client.create(self._lock_dataset_name)

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
