"""
Classes and utility functions for GCSWorkspace and GCSStepCache.
"""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import gcsfs

from tango.common.aliases import PathOrStr
from tango.common.remote_utils import (
    RemoteClient,
    RemoteConstants,
    RemoteDataset,
    RemoteDatasetConflict,
    RemoteDatasetNotFound,
    RemoteDatasetWriteError,
    RemoteFileInfo,
    RemoteStepLock,
)
from tango.step import Step
from tango.step_info import StepInfo

logger = logging.getLogger(__name__)


def empty_bucket(bucket_name: str, token: str = "google_default"):
    """
    Utility function for testing.
    """
    fs = gcsfs.GCSFileSystem(token=token)
    try:
        fs.rm(f"{bucket_name}/tango-*", recursive=True)
    except FileNotFoundError:
        pass


class GCSDataset(RemoteDataset):
    pass


@dataclass
class FileInfo(RemoteFileInfo):
    pass


class GCSClient(RemoteClient):
    """
    A client for interacting with Google Cloud Storage.
    """

    placeholder_file = ".placeholder"
    """
    The placeholder file is used for creation of a folder in the cloud bucket,
    as empty folders are not allowed. It is also used as a marker for the creation
    time of the folder, hence we use a separate file to mark the dataset as
    uncommitted.
    """

    uncommitted_file = ".uncommitted"
    """
    The uncommitted file is used to denote a dataset under construction.
    """

    settings_file = "settings.json"

    def __init__(self, bucket_name: str, token: str = "google_default"):
        # "Bucket names reside in a single namespace that is shared by all Cloud Storage users" from
        # https://cloud.google.com/storage/docs/buckets. So, the project name does not matter.
        self.gcs_fs = gcsfs.GCSFileSystem(token=token)
        self.bucket_name = bucket_name

        settings_path = self.url(self.settings_file)
        try:
            with self.gcs_fs.open(settings_path, "r") as file_ref:
                json.load(file_ref)
        except FileNotFoundError:
            settings = {"version": 1}
            with self.gcs_fs.open(settings_path, "w") as file_ref:
                json.dump(settings, file_ref)

    def url(self, dataset: Optional[str] = None):
        path = f"gs://{self.bucket_name}"
        if dataset is not None:
            path = f"{path}/{dataset}"
        return path

    @classmethod
    def _convert_ls_info_to_dataset(cls, ls_info: List[Dict]) -> GCSDataset:
        name: str
        dataset_path: str
        created: datetime
        committed: bool = True

        for info in ls_info:
            if "kind" in info and info["name"].endswith(cls.placeholder_file):
                created = datetime.strptime(info["timeCreated"], "%Y-%m-%dT%H:%M:%S.%fZ")
                dataset_path = info["name"].replace("/" + cls.placeholder_file, "")
                name = dataset_path.replace(info["bucket"] + "/", "")
            elif "kind" in info and info["name"].endswith(cls.uncommitted_file):
                committed = False

        assert name is not None, "Folder is not a GCSDataset, should not have happened."
        return GCSDataset(name, dataset_path, created, committed)

    @classmethod
    def from_env(cls, bucket_name: str):
        return cls(bucket_name, token="google_default")

    def get(self, dataset: Union[str, GCSDataset]) -> GCSDataset:
        if isinstance(dataset, str):
            path = os.path.join(self.bucket_name, dataset)
        else:
            # We have a dataset, and we recreate it with refreshed info.
            path = dataset.dataset_path
        try:
            return self._convert_ls_info_to_dataset(self.gcs_fs.ls(path=path, detail=True))
        except FileNotFoundError:
            raise RemoteDatasetNotFound()

    def create(self, dataset: str):
        # since empty folders cannot exist by themselves.
        folder_path = os.path.join(self.bucket_name, dataset)
        try:
            info = self.gcs_fs.info(folder_path)
            if info["type"] == "directory":
                raise RemoteDatasetConflict(f"{folder_path} already exists!")
            else:
                # Hack. Technically, this means that a folder of the name doesn't exist.
                # A file may still exist. Ideally, shouldn't happen.
                raise FileNotFoundError
        except FileNotFoundError:
            # Additional check for safety
            self.gcs_fs.invalidate_cache(folder_path)
            if self.gcs_fs.exists(os.path.join(folder_path, self.placeholder_file)):
                raise RemoteDatasetConflict(f"{folder_path} already exists!")
            self.gcs_fs.touch(os.path.join(folder_path, self.placeholder_file), truncate=False)
            self.gcs_fs.touch(os.path.join(folder_path, self.uncommitted_file), truncate=False)

        return self._convert_ls_info_to_dataset(self.gcs_fs.ls(folder_path, detail=True))

    def delete(self, dataset: GCSDataset):
        self.gcs_fs.rm(dataset.dataset_path, recursive=True)

    def sync(self, dataset: Union[str, GCSDataset], objects_dir: Path):
        if isinstance(dataset, str):
            folder_path = os.path.join(self.bucket_name, dataset)
        else:
            folder_path = dataset.dataset_path
        try:
            source = str(objects_dir)
            self.gcs_fs.put(source, folder_path, recursive=True)
            if objects_dir.is_dir():
                self.gcs_fs.mv(
                    os.path.join(folder_path, os.path.basename(source)) + "/*",
                    folder_path,
                    recursive=True,
                )
        except Exception:
            raise RemoteDatasetWriteError()

    def commit(self, dataset: Union[str, GCSDataset]):
        if isinstance(dataset, str):
            folder_path = os.path.join(self.bucket_name, dataset)
        else:
            folder_path = dataset.dataset_path
        uncommitted = os.path.join(folder_path, self.uncommitted_file)
        try:
            self.gcs_fs.rm_file(uncommitted)
        except FileNotFoundError:
            if not self.gcs_fs.isdir(folder_path):
                raise RemoteDatasetNotFound()
            # Otherwise, already committed. No change.

    def upload(self, dataset: GCSDataset, source: bytes, target: PathOrStr) -> None:
        file_path = os.path.join(dataset.dataset_path, target)
        with self.gcs_fs.open(file_path, "wb") as file_ref:
            file_ref.write(source)

    def get_file(self, dataset: GCSDataset, file_path: Union[str, FileInfo]):
        if isinstance(file_path, FileInfo):
            full_file_path = file_path.path
        else:
            full_file_path = os.path.join(dataset.dataset_path, file_path)
        with self.gcs_fs.open(full_file_path, "r") as file_ref:
            file_contents = file_ref.read()
        return file_contents

    def file_info(self, dataset: GCSDataset, file_path: str) -> FileInfo:
        full_file_path = os.path.join(dataset.dataset_path, file_path)
        info = self.gcs_fs.ls(full_file_path, detail=True)[0]
        return FileInfo(
            path=full_file_path,
            digest=info["md5Hash"],
            updated=datetime.strptime(info["updated"], "%Y-%m-%dT%H:%M:%S.%fZ"),
            size=info["size"],
        )

    def fetch(self, dataset: GCSDataset, target_dir: PathOrStr):
        try:
            self.gcs_fs.get(dataset.dataset_path, target_dir, recursive=True)
        except FileNotFoundError:
            raise RemoteDatasetNotFound()

    def _datasets(self, match: str, committed: bool = False) -> List[GCSDataset]:
        list_of_datasets = []
        for path in self.gcs_fs.glob(os.path.join(self.bucket_name, match) + "*"):
            info = self.gcs_fs.ls(path=path, detail=True)
            dataset = self._convert_ls_info_to_dataset(info)
            if committed:
                if not dataset.committed:
                    continue
            list_of_datasets.append(dataset)
        return list_of_datasets

    def list_steps(self, match: str) -> List[GCSDataset]:
        return self._datasets(match, committed=True)

    def list_runs(self, match: str) -> List[GCSDataset]:
        return self._datasets(match, committed=False)


def get_client(gcs_workspace: str, token: str = "google_default", **kwargs) -> GCSClient:
    # BeakerExecutor will use GOOGLE_TOKEN
    token = os.environ.get("GOOGLE_TOKEN", token)
    try:
        # If credentials dict has been passed as the token
        token = json.loads(token)
    except json.decoder.JSONDecodeError:
        pass  # It is not a json string.
    return GCSClient(gcs_workspace, token=token, **kwargs)


class Constants(RemoteConstants):
    pass


class GCSStepLock(RemoteStepLock):
    """
    Google Cloud offers consistency https://cloud.google.com/storage/docs/consistency,
    so we can use lock files.
    """

    def __init__(self, client, step: Union[str, StepInfo, Step]):
        super().__init__(client, step)
