import urllib
from pathlib import Path
from typing import List, Optional, Union

from beaker import Beaker
from beaker import Dataset as BeakerDataset
from beaker import DatasetConflict, DatasetNotFound, DatasetWriteError, FileInfo

from tango.common.aliases import PathOrStr
from tango.common.exceptions import TangoError
from tango.common.remote_utils import (
    RemoteClient,
    RemoteDatasetConflict,
    RemoteDatasetNotFound,
)
from tango.version import VERSION

# from tango.common.util import utc_now_datetime


class BeakerDatasetNotFound(RemoteDatasetNotFound):
    # TODO: we likely don't need this empty abstraction. Can just raise RemoteDatasetNotFound.
    pass


class BeakerDatasetConflict(RemoteDatasetConflict):
    pass


class BeakerDatasetWriteError(TangoError):
    pass


class BeakerClient(RemoteClient):
    """
    A client for interacting with beaker.
    TODO: this may or may not be the best way of dealing with the remote client abstraction.
    Feedback needed.
    """

    def __init__(self, beaker_workspace: Optional[str] = None, **kwargs):
        user_agent = f"tango v{VERSION}"
        if beaker_workspace is not None:
            self.beaker = Beaker.from_env(
                default_workspace=beaker_workspace,
                session=True,
                user_agent=user_agent,
                **kwargs,
            )
        else:
            self.beaker = Beaker.from_env(session=True, user_agent=user_agent, **kwargs)

    @property
    def full_name(self):
        return self.beaker.workspace.get().full_name

    def url(self, dataset: Optional[str] = None):
        return self.beaker.ur

    def dataset_url(self, workspace_url: str, dataset_name: str) -> str:
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

    def get(self, dataset: Union[str, BeakerDataset]) -> BeakerDataset:
        try:
            return self.beaker.dataset.get(dataset)
        except DatasetConflict:
            # We do this so that remote_workspace gets errors of the type RemoteDatasetNotFound.
            raise BeakerDatasetNotFound()

    def create(self, dataset: str, commit: bool = False):
        try:
            self.beaker.dataset.create(dataset, commit=commit)
        except DatasetConflict:
            raise BeakerDatasetConflict()

    def delete(self, dataset: BeakerDataset):
        self.beaker.dataset.delete(dataset)

    def sync(self, dataset: Union[str, BeakerDataset], objects_dir: Path):
        try:
            self.beaker.dataset.sync(dataset, objects_dir, quiet=True)
        except DatasetWriteError:
            raise BeakerDatasetWriteError()

    def commit(self, dataset: Union[str, BeakerDataset]):
        self.beaker.dataset.commit(dataset)

    def upload(self, dataset: BeakerDataset, source: bytes, target: PathOrStr) -> None:
        self.beaker.dataset.upload(dataset, source, target)

    def get_file(self, dataset: BeakerDataset, file_path: Union[str, FileInfo]):
        try:
            self.beaker.dataset.get_file(dataset, file_path, quiet=True)
        except DatasetNotFound:
            raise BeakerDatasetNotFound()

    def file_info(self, dataset: BeakerDataset, file_path: str) -> FileInfo:
        try:
            return self.beaker.dataset.file_info(dataset, file_path)
        except DatasetNotFound:
            raise BeakerDatasetNotFound()

    def fetch(self, dataset: BeakerDataset, target_dir: PathOrStr):
        try:
            self.beaker.dataset.fetch(dataset, target_dir, quiet=True)
        except DatasetNotFound:
            raise BeakerDatasetNotFound()

    def datasets(
        self, match: str, uncommitted: bool = True, results: bool = False
    ) -> List[BeakerDataset]:
        return self.beaker.workspace.iter_datasets(
            match=match, uncommitted=uncommitted, results=results
        )
