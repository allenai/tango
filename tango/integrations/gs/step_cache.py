import logging
from pathlib import Path
from typing import Optional, Union

from google.cloud import storage

from tango.common.aliases import PathOrStr
from tango.common.remote_utils import (
    RemoteDatasetConflict,
    RemoteDatasetNotFound,
    RemoteDatasetWriteError,
)
from tango.common.util import make_safe_filename, tango_cache_dir
from tango.step import Step
from tango.step_cache import StepCache
from tango.step_caches.remote_step_cache import RemoteStepCache
from tango.step_info import StepInfo

from .common import Constants, GCSClient, GCSDataset

logger = logging.getLogger(__name__)


@StepCache.register("gs")
class GSStepCache(RemoteStepCache):
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

    def __init__(self, bucket_name: str, client: Optional[GCSClient] = None):
        if client is not None:
            assert (
                bucket_name == client.bucket_name
            ), "Assert that bucket name is same as client bucket until we do better"
            self.bucket_name = bucket_name
            self.client = client
        else:
            self.client = GCSClient(bucket_name)
        super().__init__(
            tango_cache_dir() / "gs_cache" / make_safe_filename(self.client.bucket_name)
        )

    def _step_result_remote(self, step: Union[Step, StepInfo]) -> Optional[GCSDataset]:
        try:
            dataset = self.client.get(Constants.step_dataset_name(step))
            return dataset if dataset.committed else None
        except RemoteDatasetNotFound:
            return None

    def _sync_step_remote(self, step: Step, objects_dir: Path) -> storage.blob.Blob:
        dataset_name = Constants.step_dataset_name(step)

        try:
            self.client.create(dataset_name, commit=False)
        except RemoteDatasetConflict:
            pass
        try:
            dataset = self.client.sync(dataset_name, objects_dir)
            dataset = self.client.commit(dataset)
        except RemoteDatasetWriteError:
            pass

        return dataset

    def _fetch_step_remote(self, step_result, target_dir: PathOrStr):
        try:
            self.client.fetch(step_result, target_dir)
        except RemoteDatasetNotFound:
            self._raise_remote_not_found()

    def __len__(self):
        # NOTE: lock datasets should not count here. They start with the same prefix,
        # but they never get committed.
        return sum(
            1
            for ds in self.client.datasets(uncommitted=False, match=Constants.STEP_DATASET_PREFIX)
            if ds.name is not None and ds.name.startswith(Constants.STEP_DATASET_PREFIX)
        )
