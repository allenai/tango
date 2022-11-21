import datetime
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import gcsfs
from google.cloud import storage

from tango.common.aliases import PathOrStr
from tango.common.exceptions import TangoError

_CREDENTIALS_WARNING_ISSUED = False


def check_environment():
    global _CREDENTIALS_WARNING_ISSUED
    if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ and not _CREDENTIALS_WARNING_ISSUED:
        warnings.warn(
            "Missing environment variable 'GOOGLE_APPLICATION_CREDENTIALS' required to authenticate "
            "to Weights & Biases.",
            UserWarning,
        )
        _CREDENTIALS_WARNING_ISSUED = True


# TODO: remove this later.
class CloudStorageWrapper:

    # TODO: add various try/excepts to make it robust.
    # TODO: can potentially generalize wrapper to any remote workspace client.
    def __init__(self, storage_client: Optional[storage.Client] = None):
        check_environment()
        if storage_client is not None:
            self.storage_client = storage_client
        else:
            self.storage_client = storage.Client()

    def _create_bucket(self, bucket_name: str, location: str = "US") -> storage.Bucket:
        bucket = self.storage_client.bucket(bucket_name)
        bucket.create(location=location)
        return bucket

    def _get_bucket(self, bucket_name: str) -> storage.Bucket:
        bucket = self.storage_client.get_bucket(bucket_name)
        return bucket

    def _upload_file_to_bucket(self, bucket_name: str, blob_name: str, file_path: str) -> None:
        bucket = self._get_bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(file_path)

    def _download_file_from_bucket(self, bucket_name: str, blob_name: str, file_path: str) -> None:
        bucket = self._get_bucket(bucket_name)
        blob = bucket.blob(blob_name)
        with open(file_path, "wb") as file_ref:
            self.storage_client.download_blob_to_file(blob, file_ref)

    def _upload_dir_to_bucket(self, bucket_name: str, dir_path: Path):
        # Can do it with gcsfs. Also, using it will offload some of the auth stuff.
        pass


class GCSDatasetNotFound(TangoError):
    pass


class GCSDatasetConflict(TangoError):
    pass


class GCSDatasetWriteError(TangoError):
    pass


class GCSDataset:
    # For the purpose of this, GCSDataset will be conceptually a folder.
    def __init__(self, info: Dict):
        # info here is just what gcs_fs returns.
        self.info = info
        self.committed = True  # TODO: Do some other check (maybe presence of a file?)
        self.created = datetime.datetime.now()  # TODO: read the updated time from the object!

    def dataset_path(self):
        # Includes name of bucket.
        return self.info["name"]

    @property
    def name(self):
        return self.info["name"].replace(self.info["bucket"] + "/", "")


@dataclass
class FileInfo:
    # TODO: this is just mirroring beaker right now. We may not need this level of abstraction.
    path: str
    digest: str
    updated: str  # TODO: convert to datetime.
    size: int


class GCSClient:
    """
    A client for interacting with Google Cloud Storage.
    """

    placeholder_file = ".placeholder"

    def __init__(self, bucket_name: str, token: str = "google_default"):
        # "Bucket names reside in a single namespace that is shared by all Cloud Storage users" from
        # https://cloud.google.com/storage/docs/buckets. So, the project name does not matter. TODO: Confirm.
        self.gcs_fs = gcsfs.GCSFileSystem(token=token)
        self.bucket_name = bucket_name

    @classmethod
    def from_env(cls, bucket_name: str):
        return cls(bucket_name, token="google_default")

    def url(self, dataset: Optional[str] = None):
        path = f"gs://{self.bucket_name}"
        if dataset:
            path = f"{path}/dataset"
        return path

    @property
    def full_name(self):
        return self.bucket_name

    def get(self, dataset: Union[str, GCSDataset]) -> GCSDataset:
        if isinstance(dataset, str):
            path = os.path.join(self.bucket_name, dataset)
        else:
            # We have a dataset, and we recreate it with refreshed info.
            path = dataset.dataset_path()
        try:
            return GCSDataset(self.gcs_fs.info(path=path))
        except FileNotFoundError:
            raise GCSDatasetNotFound()

    def create(self, dataset: str, commit: bool = False):
        # TODO: commit will always be False for this creation
        # since empty folders cannot exist by themselves.
        folder_path = os.path.join(self.bucket_name, dataset)
        try:
            info = self.gcs_fs.info(folder_path)
            if info["type"] == "directory":
                raise GCSDatasetConflict(f"{folder_path} already exists!")
            else:
                # Hack. Technically, this means that a folder of the name doesn't exist.
                # A file may still exist. Ideally, shouldn't happen.
                raise FileNotFoundError
        except FileNotFoundError:
            with self.gcs_fs.open(
                os.path.join(folder_path, self.placeholder_file), "w"
            ) as file_ref:
                file_ref.write("placeholder")

        return GCSDataset(self.gcs_fs.info(folder_path))

    def delete(self, dataset: GCSDataset):
        self.gcs_fs.rm(dataset.dataset_path(), recursive=True)

    def sync(self, dataset: Union[str, GCSDataset], objects_dir: Path):
        if isinstance(dataset, str):
            folder_path = os.path.join(self.bucket_name, dataset)
        else:
            folder_path = dataset.dataset_path()
        # Does not remove files that may have been present before, but aren't now.
        # TODO: potentially consider gsutil rsync with --delete --ignore-existing command
        # Using gsutil programmatically:
        # https://github.com/GoogleCloudPlatform/gsutil/blob/84aa9af730fe3fa1307acc1ab95aec684d127152/gslib/tests/test_rsync.py
        # TODO: if not using rsync, then do simple checks.
        self.gcs_fs.put(str(objects_dir), folder_path, recursive=True)
        return self.get(dataset)

    def upload(self, dataset: GCSDataset, source: bytes, target: PathOrStr) -> None:
        file_path = os.path.join(dataset.dataset_path(), target)
        with self.gcs_fs.open(file_path, "wb") as file_ref:
            file_ref.write(source)

    def get_file(self, dataset: GCSDataset, file_path: Union[str, FileInfo]):
        if isinstance(file_path, FileInfo):
            full_file_path = file_path.path
        else:
            full_file_path = os.path.join(dataset.dataset_path(), file_path)
        with self.gcs_fs.open(full_file_path, "r") as file_ref:
            file_contents = file_ref.read()
        return file_contents

    def file_info(self, dataset: GCSDataset, file_path: str) -> FileInfo:
        full_file_path = os.path.join(dataset.dataset_path(), file_path)
        info = self.gcs_fs.ls(full_file_path, detail=True)[0]  # TODO: add sanity checks
        # TODO: should digest be crc32c instead of md5Hash?
        return FileInfo(
            path=full_file_path, digest=info["md5Hash"], updated=info["updated"], size=info["size"]
        )

    def fetch(self, dataset: GCSDataset, target_dir: PathOrStr):
        try:
            self.gcs_fs.get(dataset.dataset_path(), target_dir)
        except FileNotFoundError:
            raise GCSDatasetNotFound()

    def datasets(
        self, match: str, uncommitted: bool = False, results: bool = False
    ) -> List[GCSDataset]:
        # TODO: do we need the concept of committed?
        # TODO: what do we do with results?
        list_of_datasets = []
        for path in self.gcs_fs.glob(os.path.join(self.bucket_name, match) + "*"):
            info = self.gcs_fs.info(path=path)
            if info["type"] == "directory":
                list_of_datasets.append(GCSDataset(info))
        return list_of_datasets
