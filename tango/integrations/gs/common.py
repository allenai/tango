"""
Classes and utility functions for GCSWorkspace and GCSStepCache.
"""

import glob
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
    uncommitted_file = ".uncommitted"

    def __init__(self, bucket_name: str, token: str = "google_default"):
        # "Bucket names reside in a single namespace that is shared by all Cloud Storage users" from
        # https://cloud.google.com/storage/docs/buckets. So, the project name does not matter.
        self.gcs_fs = gcsfs.GCSFileSystem(token=token)
        self.bucket_name = bucket_name

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

    @property
    def full_name(self):
        return self.bucket_name

    @classmethod
    def dataset_url(cls, workspace_url: str, dataset_name: str) -> str:
        return os.path.join(workspace_url, dataset_name)

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

    def create(self, dataset: str, commit: bool = False):
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
            self.gcs_fs.touch(os.path.join(folder_path, self.placeholder_file), truncate=False)
            if not commit:
                self.gcs_fs.touch(os.path.join(folder_path, self.uncommitted_file), truncate=False)

        return self._convert_ls_info_to_dataset(self.gcs_fs.ls(folder_path, detail=True))

    def delete(self, dataset: GCSDataset):
        self.gcs_fs.rm(dataset.dataset_path, recursive=True)

    def sync(self, dataset: Union[str, GCSDataset], objects_dir: Path):
        if isinstance(dataset, str):
            folder_path = os.path.join(self.bucket_name, dataset)
        else:
            folder_path = dataset.dataset_path
        # Does not remove files that may have been present before, but aren't now.
        # TODO: potentially consider gsutil rsync with --delete --ignore-existing command
        # Using gsutil programmatically:
        # https://github.com/GoogleCloudPlatform/gsutil/blob/84aa9af730fe3fa1307acc1ab95aec684d127152/gslib/tests/test_rsync.py
        try:
            source = str(objects_dir)
            if objects_dir.is_dir():
                source += "/*"
            for path in glob.glob(source):
                basepath = os.path.basename(path)
                self.gcs_fs.put(path, os.path.join(folder_path, basepath), recursive=True)
            # The put command below seems to have inconsistent results at the top level.
            # TODO: debug later.
            # self.gcs_fs.put(source, folder_path, recursive=True)
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
            # Already committed. No change.
            pass

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
        # TODO: should digest be crc32c instead of md5Hash?
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

    def datasets(self, match: str, uncommitted: bool = True) -> List[GCSDataset]:
        list_of_datasets = []
        for path in self.gcs_fs.glob(os.path.join(self.bucket_name, match) + "*"):
            info = self.gcs_fs.ls(path=path, detail=True)
            dataset = self._convert_ls_info_to_dataset(info)
            if not uncommitted:
                if not dataset.committed:
                    continue
            list_of_datasets.append(dataset)
        return list_of_datasets


def _is_json_str(string: str) -> bool:
    return "{" in string


def get_client(gcs_workspace: str, token: str = "google_default", **kwargs) -> GCSClient:
    # BeakerExecutor will use GOOGLE_TOKEN
    token = os.environ.get("GOOGLE_TOKEN", token)
    if _is_json_str(token):
        token = json.loads(token)
    return GCSClient(gcs_workspace, token=token, **kwargs)


class Constants(RemoteConstants):
    pass


class GCSStepLock(RemoteStepLock):
    def __init__(self, client, step: Union[str, StepInfo, Step]):
        super().__init__(client, step)
