import os
import warnings
from pathlib import Path
from typing import Optional

from google.cloud import storage

_CREDENTIALS_WARNING_ISSUED = False


def check_environment():
    global _CREDENTIALS_WARNING_ISSUED
    if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ and not _CREDENTIALS_WARNING_ISSUED:
        warnings.warn(
            "Missing environment variable 'GOOGLE_APPLICATION_CREDENTIALS' required to authenticate to Weights & Biases.",
            UserWarning,
        )
        _CREDENTIALS_WARNING_ISSUED = True


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
