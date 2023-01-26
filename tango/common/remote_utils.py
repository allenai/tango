import datetime
import logging
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional, Union

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
    Abstraction for all objects in remote storage locations. Conceptually, this can be thought of as a folder.
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
    committed: bool
    """
    If set to True, no further changes to the dataset are allowed.
    If set to False, it means that the dataset is under construction.
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
        Creates a new dataset in the remote location. By default, it is uncommitted.
        """
        raise NotImplementedError()

    @abstractmethod
    def delete(self, dataset):
        """
        Removes a dataset from the remote location.
        """
        raise NotImplementedError()

    @abstractmethod
    def upload(self, dataset, objects_dir):
        """
        Writes the contents of objects_dir to the remote dataset location.
        """
        raise NotImplementedError()

    @abstractmethod
    def commit(self, dataset):
        """
        Marks the dataset as committed. No further changes to the dataset are allowed.
        """
        raise NotImplementedError()

    @abstractmethod
    def download(self, dataset, target_dir: PathOrStr) -> RemoteDataset:
        """
        Writes the contents of the remote dataset to the `target_dir`.
        """
        raise NotImplementedError()

    @abstractmethod
    def datasets(self, match: str, uncommitted: bool = True):
        """
        Lists all the datasets within the remote storage, based on
        `match` and `uncommitted` criteria. These can include steps and runs.
        """
        raise NotImplementedError()
