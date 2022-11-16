import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional

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


class GCSDataset:
    def __init__(self, info: Dict):
        self.info = info
        self.committed = True  # TODO: Do some other check (maybe presence of a file?)

    def dataset_path(self):
        # Includes name of bucket.
        return self.info["name"]

    @property
    def name(self):
        return self.info["name"].replace(self.info["bucket"] + "/", "")


class GCSDatasetNotFound(TangoError):
    pass


class GCSClient:
    """
    A client for interacting with Google Cloud Storage.
    """

    def __init__(self, bucket_name: str, project: Optional[str] = None, location: str = "US"):
        # TODO: with google_default, cannot specify project.
        # TODO: FIX Credentials are inferred from the environment right now.
        self.gcs_fs = gcsfs.GCSFileSystem(token="google_default")
        self.bucket_name = bucket_name

    def get(self, dataset: str) -> GCSDataset:
        path = os.path.join(self.bucket_name, dataset)
        try:
            return GCSDataset(self.gcs_fs.info(path=path))
        except FileNotFoundError:
            raise GCSDatasetNotFound()

    def sync(self, dataset: str, objects_dir: Path):
        path = os.path.join(self.bucket_name, dataset)
        # Does not remove files that may have been present before, but aren't now.
        # TODO: potentially consider gsutil rsync with --delete --ignore-existing command
        # Using gsutil programmatically:
        # https://github.com/GoogleCloudPlatform/gsutil/blob/84aa9af730fe3fa1307acc1ab95aec684d127152/gslib/tests/test_rsync.py

        self.gcs_fs.put(str(objects_dir), path, recursive=True)
        return self.get(dataset)

    def fetch(self, dataset: GCSDataset, target_dir: PathOrStr):
        try:
            self.gcs_fs.get(dataset.dataset_path(), target_dir)
        except FileNotFoundError:
            raise GCSDatasetNotFound()

    def datasets(self, match: str, uncommitted: bool = False) -> List[GCSDataset]:
        # TODO: do we need the concept of committed?
        list_of_datasets = []
        for path in self.gcs_fs.glob(os.path.join(self.bucket_name, match) + "*"):
            info = self.gcs_fs.info(path=path)
            if info["type"] == "directory":
                list_of_datasets.append(GCSDataset(info))
        return list_of_datasets
