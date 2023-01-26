"""
Classes and utility functions for GSWorkspace and GSStepCache.
"""
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import google.auth
from google.api_core import exceptions
from google.cloud import storage
from google.oauth2.credentials import Credentials

from tango.common.aliases import PathOrStr
from tango.common.remote_utils import (
    RemoteClient,
    RemoteConstants,
    RemoteDataset,
    RemoteDatasetConflict,
    RemoteDatasetNotFound,
    RemoteDatasetWriteError,
    RemoteStepLock,
)
from tango.step import Step
from tango.step_info import StepInfo

logger = logging.getLogger(__name__)


def empty_bucket(bucket_name: str):
    """
    Utility function for testing.
    """
    credentials, project = google.auth.default()
    client = storage.Client(project=project, credentials=credentials)
    bucket = client.bucket(bucket_name)
    try:
        bucket.delete_blobs(list(bucket.list_blobs(prefix="tango-")))
    except exceptions.NotFound:
        pass


def empty_datastore(namespace: str):
    """
    Utility funtion for testing.
    """
    from google.cloud import datastore

    credentials, project = google.auth.default()
    client = datastore.Client(project=project, credentials=credentials, namespace=namespace)
    run_query = client.query(kind="run")
    run_query.keys_only()
    keys = [entity.key for entity in run_query.fetch()]
    stepinfo_query = client.query(kind="stepinfo")
    stepinfo_query.keys_only()
    keys += [entity.key for entity in stepinfo_query.fetch()]
    client.delete_multi(keys)


class GCSDataset(RemoteDataset):
    """
    A GCSDataset object is used for representing storage objects in google cloud storage.
    Currently, it can be either a cached Step, or a Run.
    """

    pass


class GSStep(GCSDataset):
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

    def __init__(
        self,
        bucket_name: str,
        credentials: Optional[Credentials] = None,
        project: Optional[str] = None,
    ):
        # https://googleapis.dev/python/google-auth/latest/user-guide.html#service-account-private-key-files
        # https://googleapis.dev/python/google-api-core/latest/auth.html
        if not credentials:
            credentials, project = google.auth.default()

        self.storage = storage.Client(project=project, credentials=credentials)
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
        return cls(bucket_name)

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

    def upload(self, dataset: Union[str, GCSDataset], objects_dir: Path):
        if isinstance(dataset, str):
            folder_path = dataset
        else:
            folder_path = dataset.dataset_path

        source_path = str(objects_dir)

        def _sync_blob(source_file_path: str, target_file_path: str):
            blob = self.storage.bucket(self.bucket_name).blob(target_file_path)
            blob.upload_from_filename(source_file_path)

        try:
            if os.path.isdir(source_path):
                for dirpath, _, filenames in os.walk(source_path):
                    for filename in filenames:
                        # TODO: CI fails with ThreadPoolExecutor parallelism. Debug later.
                        source_file_path = os.path.join(dirpath, filename)
                        target_file_path = os.path.join(
                            folder_path, source_file_path.replace(source_path + "/", "")
                        )
                        _sync_blob(source_file_path, target_file_path)
            else:
                source_file_path = source_path
                target_file_path = os.path.join(folder_path, os.path.basename(source_file_path))
                _sync_blob(source_file_path, target_file_path)

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

    def download(self, dataset: GCSDataset, target_dir: PathOrStr):
        assert (
            self.storage.bucket(self.bucket_name)
            .blob(os.path.join(dataset.dataset_path, self.placeholder_file))
            .exists()
        )

        def _fetch_blob(blob: storage.Blob):
            blob.update()  # fetches updated information.
            source_path = blob.name.replace(dataset.dataset_path + "/", "")
            target_path = os.path.join(target_dir, source_path)
            if not os.path.exists(os.path.dirname(target_path)):
                os.mkdir(os.path.dirname(target_path))
            blob.download_to_filename(target_path)

        try:
            for blob in self.storage.list_blobs(self.bucket_name, prefix=dataset.dataset_path):
                _fetch_blob(blob)
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


def get_credentials(credentials: Optional[Union[str, Credentials]] = None):
    # BeakerExecutor will use GOOGLE_TOKEN
    credentials = os.environ.get("GOOGLE_TOKEN", credentials)
    if credentials is not None:
        if isinstance(credentials, str) and credentials.endswith(".json"):
            with open(credentials) as file_ref:
                credentials = file_ref.read()
        try:
            # If credentials dict has been passed as a json string
            credentials_dict = json.loads(credentials)
            credentials_dict.pop("type", None)
            # sometimes the credentials dict may not contain `token` key, but `Credentials()` needs the parameter.
            token = credentials_dict.pop("token", None)
            token_uri = credentials_dict.pop("token_uri", "https://oauth2.googleapis.com/token")
            credentials = Credentials(token=token, token_uri=token_uri, **credentials_dict)
        except json.decoder.JSONDecodeError:
            # It is not a json string.
            # We do this because BeakerExecutor cannot write a None secret.
            if credentials == "default":
                credentials = None
    if not credentials:
        credentials, _ = google.auth.default()
    return credentials


def get_client(
    gcs_workspace: str, credentials: Optional[Union[str, Credentials]] = None, **kwargs
) -> GCSClient:
    credentials = get_credentials(credentials)
    return GCSClient(gcs_workspace, credentials=credentials, **kwargs)


class Constants(RemoteConstants):
    pass


class GCSStepLock(RemoteStepLock):
    """
    Google Cloud offers consistency https://cloud.google.com/storage/docs/consistency,
    so we can use lock files.
    """

    def __init__(self, client, step: Union[str, StepInfo, Step]):
        super().__init__(client, step)