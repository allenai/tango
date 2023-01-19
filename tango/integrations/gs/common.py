"""
Classes and utility functions for GCSWorkspace and GCSStepCache.
"""
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

from google.api_core import exceptions
from google.cloud import storage

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
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    try:
        bucket.delete_blobs(list(bucket.list_blobs(prefix="tango-")))
    except exceptions.NotFound:
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
    """
    This file is for storing metadata like version information, etc.
    """

    def __init__(self, bucket_name: str, token: str = "google_default"):
        # https://googleapis.dev/python/google-auth/latest/user-guide.html#service-account-private-key-files
        # https://googleapis.dev/python/google-api-core/latest/auth.html
        self.storage = storage.Client()  # TODO: use oauth2 credentials.
        self.bucket_name = bucket_name

        blob = self.storage.bucket(bucket_name).blob(self.settings_file)  # no HTTP request yet
        try:
            with blob.open("r") as file_ref:
                json.load(file_ref)
        except exceptions.NotFound:
            settings = {"version": 1}
            with blob.open("w") as file_ref:
                json.dump(settings, file_ref)

    def url(self, dataset: Optional[str] = None):
        path = f"gs://{self.bucket_name}"
        if dataset is not None:
            path = f"{path}/{dataset}"
        return path

    @classmethod
    def _convert_blobs_to_dataset(cls, blobs: List[storage.Blob]) -> GCSDataset:
        name: str
        dataset_path: str
        created: datetime
        committed: bool = True

        for blob in blobs:
            if blob.name.endswith(cls.placeholder_file):
                created = blob.time_created
                name = blob.name.replace("/" + cls.placeholder_file, "")
                dataset_path = name  # does not contain bucket info here.
            elif blob.name.endswith(cls.uncommitted_file):
                committed = False

        assert name is not None, "Folder is not a GCSDataset, should not have happened."
        return GCSDataset(name, dataset_path, created, committed)

    @classmethod
    def from_env(cls, bucket_name: str):
        return cls(bucket_name, token="google_default")

    def get(self, dataset: Union[str, GCSDataset]) -> GCSDataset:
        if isinstance(dataset, str):
            path = dataset
        else:
            # We have a dataset, and we recreate it with refreshed info.
            path = dataset.dataset_path

        blobs = list(self.storage.bucket(self.bucket_name).list_blobs(prefix=path))
        if len(blobs) > 0:
            return self._convert_blobs_to_dataset(blobs)
        else:
            raise RemoteDatasetNotFound()

    def create(self, dataset: str):
        bucket = self.storage.bucket(self.bucket_name)
        # gives refreshed information
        if bucket.blob(os.path.join(dataset, self.placeholder_file)).exists():
            raise RemoteDatasetConflict(f"{dataset} already exists!")
        else:
            # Additional safety check
            if bucket.blob(os.path.join(dataset, self.uncommitted_file)).exists():
                raise RemoteDatasetConflict(f"{dataset} already exists!")
            bucket.blob(os.path.join(dataset, self.placeholder_file)).upload_from_string("")
            bucket.blob(os.path.join(dataset, self.uncommitted_file)).upload_from_string("")
        return self._convert_blobs_to_dataset(list(bucket.list_blobs(prefix=dataset)))

    def delete(self, dataset: GCSDataset):
        bucket = self.storage.bucket(self.bucket_name)
        blobs = list(bucket.list_blobs(prefix=dataset.dataset_path))
        bucket.delete_blobs(blobs)

    def sync(self, dataset: Union[str, GCSDataset], objects_dir: Path):
        if isinstance(dataset, str):
            folder_path = dataset
        else:
            folder_path = dataset.dataset_path

        source_path = str(objects_dir)
        # TODO: this is copying one file at a time, may want to parallelize.
        try:
            for dirpath, _, filenames in os.walk(source_path):
                for filename in filenames:
                    source_file_path = os.path.join(dirpath, filename)
                    target_file_path = os.path.join(
                        folder_path, source_file_path.replace(source_path + "/", "")
                    )
                    blob = self.storage.bucket(self.bucket_name).blob(target_file_path)
                    blob.upload_from_filename(source_file_path)
        except Exception:
            raise RemoteDatasetWriteError()

    def commit(self, dataset: Union[str, GCSDataset]):
        if isinstance(dataset, str):
            folder_path = dataset
        else:
            folder_path = dataset.dataset_path
        bucket = self.storage.bucket(self.bucket_name)
        try:
            bucket.delete_blob(os.path.join(folder_path, self.uncommitted_file))
        except exceptions.NotFound:
            if not bucket.blob(os.path.join(folder_path, self.placeholder_file)).exists():
                raise RemoteDatasetNotFound()
            # Otherwise, already committed. No change.

    def upload(self, dataset: GCSDataset, source: bytes, target: PathOrStr) -> None:
        file_path = os.path.join(dataset.dataset_path, target)
        blob = self.storage.bucket(self.bucket_name).blob(file_path)
        with blob.open("wb") as file_ref:
            file_ref.write(source)

    def get_file(self, dataset: GCSDataset, file_path: Union[str, FileInfo]):
        if isinstance(file_path, FileInfo):
            full_file_path = file_path.path
        else:
            full_file_path = os.path.join(dataset.dataset_path, file_path)
        blob = self.storage.bucket(self.bucket_name).blob(full_file_path)
        with blob.open("r") as file_ref:
            file_contents = file_ref.read()
        return file_contents

    def file_info(self, dataset: GCSDataset, file_path: str) -> FileInfo:
        full_file_path = os.path.join(dataset.dataset_path, file_path)
        blob = self.storage.bucket(self.bucket_name).get_blob(
            full_file_path
        )  # get_blob makes a request
        return FileInfo(
            path=full_file_path,
            digest=blob.md5_hash,
            updated=blob.updated,
            size=blob.size,
        )

    def fetch(self, dataset: GCSDataset, target_dir: PathOrStr):
        assert (
            self.storage.bucket(self.bucket_name)
            .blob(os.path.join(dataset.dataset_path, self.placeholder_file))
            .exists()
        )
        # TODO: this is copying one file at a time, may want to parallelize.
        try:
            for blob in self.storage.list_blobs(self.bucket_name, prefix=dataset.dataset_path):
                blob.update()  # fetches updated information.
                source_path = blob.name.replace(dataset.dataset_path + "/", "")
                target_path = os.path.join(target_dir, source_path)
                if not os.path.exists(os.path.dirname(target_path)):
                    os.mkdir(os.path.dirname(target_path))
                blob.download_to_filename(target_path)
        except exceptions.NotFound:
            raise RemoteDatasetWriteError()

    def datasets(self, match: str, uncommitted: bool = True) -> List[GCSDataset]:
        list_of_datasets = []
        for folder_name in self.storage.list_blobs(
            self.bucket_name, prefix=match, delimiter="/"
        )._get_next_page_response()["prefixes"]:
            dataset = self._convert_blobs_to_dataset(
                list(self.storage.list_blobs(self.bucket_name, prefix=folder_name))
            )
            if not uncommitted:
                if not dataset.committed:
                    continue
            list_of_datasets.append(dataset)
        return list_of_datasets


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
