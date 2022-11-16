import logging
import os
from pathlib import Path
from typing import Optional, Union

from google.cloud import storage

from tango.common.aliases import PathOrStr
from tango.common.exceptions import TangoError
from tango.common.util import make_safe_filename, tango_cache_dir
from tango.step import Step
from tango.step_cache import StepCache
from tango.step_caches.remote_step_cache import RemoteConstants, RemoteStepCache
from tango.step_info import StepInfo

from .util import GCSClient, GCSDataset, GCSDatasetNotFound

logger = logging.getLogger(__name__)

# TODO: duplicate code. move some place else.


class Constants(RemoteConstants):
    pass


def step_dataset_name(step: Union[str, StepInfo, Step]) -> str:
    return f"{Constants.STEP_DATASET_PREFIX}{step if isinstance(step, str) else step.unique_id}"


def step_lock_dataset_name(step: Union[str, StepInfo, Step]) -> str:
    return f"{step_dataset_name(step)}-lock"


def run_dataset_name(name: str) -> str:
    return f"{Constants.RUN_DATASET_PREFIX}{name}"


class GCSDatasetConflict(TangoError):
    pass


class GCSDatasetWriteError(TangoError):
    pass


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
        super().__init__(tango_cache_dir() / "gcs_cache" / make_safe_filename(bucket_name))
        self.bucket_name = bucket_name
        self.gcs_client = GCSClient(bucket_name)

    def _gcs_path(self, blob_name: str):
        return os.path.join(self.bucket_name, blob_name)

    def _step_result_remote(self, step: Union[Step, StepInfo]) -> Optional[GCSDataset]:
        try:
            dataset = self.gcs_client.get(step_dataset_name(step))
            return dataset if dataset.committed is not None else None
        except GCSDatasetNotFound:
            return None

    def _sync_step_remote(self, step: Step, objects_dir: Path) -> storage.blob.Blob:
        dataset_name = step_dataset_name(step)

        try:
            dataset = self.gcs_client.sync(dataset_name, objects_dir)
            # dataset = self.gcs_client.commit(dataset)
        except GCSDatasetWriteError:  # TODO: this doesn't happen yet.
            pass

        return dataset

    def _fetch_step_remote(self, step_result, target_dir: PathOrStr):
        try:
            self.gcs_client.fetch(step_result, target_dir)
        except GCSDatasetNotFound:
            self._raise_remote_not_found()

    def __len__(self):
        # NOTE: lock datasets should not count here. They start with the same prefix,
        # but they never get committed.
        # TODO: check for lock files in a different way.
        return sum(
            1
            for ds in self.gcs_client.datasets(
                uncommitted=False, match=Constants.STEP_DATASET_PREFIX
            )
            if ds.name is not None and ds.name.startswith(Constants.STEP_DATASET_PREFIX)
        )
