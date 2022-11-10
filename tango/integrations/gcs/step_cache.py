import logging
import os
from pathlib import Path
from typing import Optional, Union

import gcsfs
from google.api_core.exceptions import NotFound
from google.cloud import storage

from tango.common.aliases import PathOrStr
from tango.common.file_lock import FileLock
from tango.common.util import tango_cache_dir
from tango.integrations.gcs.util import CloudStorageWrapper
from tango.step import Step
from tango.step_cache import CacheMetadata, StepCache
from tango.step_caches.local_step_cache import LocalStepCache
from tango.step_caches.remote_step_cache import Constants, RemoteStepCache
from tango.step_info import StepInfo

logger = logging.getLogger(__name__)


# TODO: copy. move some place else.
def step_blob_name(step: Union[str, StepInfo, Step]) -> str:
    return f"{Constants.STEP_DATASET_PREFIX}{step if isinstance(step, str) else step.unique_id}"


@StepCache.register("gcs")
class GCSStepCache(RemoteStepCache):
    """
    This is a :class:`~tango.step_cache.StepCache` that's used by :class:`GCSWorkspace`.
    It stores the results of steps on Google cloud buckets as blobs.

    It also keeps a limited in-memory cache as well as a local backup on disk, so fetching a
    step's resulting subsequent times should be fast.

    .. tip::
        Registered as a :class:`~tango.step_cache.StepCache` under the name "gcs".

    :param bucket_name: The name of the google cloud bucket to use.
    :param storage_client: The google cloud storage client to use.
    """

    def __init__(self, bucket_name: str, storage_client: Optional[storage.Client] = None):
        super().__init__(tango_cache_dir() / "gcs_cache")
        self.bucket_name = bucket_name
        self.gcs_fs = gcsfs.GCSFileSystem()  # Uses the default credentials.
        self.client_wrapper = CloudStorageWrapper()
        try:
            self.bucket = self.client_wrapper._get_bucket(bucket_name)
        except:
            self.bucket = self.client_wrapper._create_bucket(bucket_name)

    def _gcs_path(self, blob_name: str):
        return os.path.join(self.bucket_name, blob_name)

    def _step_result_remote(self, step: Union[Step, StepInfo]):
        # blob = self.bucket.get_blob(step_blob_name(step))
        # return blob if blob.exists() else None
        blob_name = step_blob_name(step)
        result_path = self._gcs_path(blob_name)
        if self.gcs_fs.exists():
            return result_path
        return None

    def _sync_step_remote(self, step: Step, objects_dir: Path) -> storage.blob.Blob:
        blob_name = step_blob_name(step)
        blob = self.bucket.blob(blob_name)
        blob.update()
        try:
            self.gcs_fs.put(objects_dir, self._gcs_path(blob_name))
        except:
            # TODO
            pass

    def _fetch_step_remote(self, step_result, target_dir: PathOrStr):
        if step_result:
            self.gcs_fs.get(step_result, target_dir)
        else:
            self._raise_remote_not_found()

    def __len__(self):
        pass
