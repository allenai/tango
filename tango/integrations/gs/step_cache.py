import logging
from pathlib import Path
from typing import Optional, Union

from tango.common import PathOrStr
from tango.common.util import make_safe_filename, tango_cache_dir
from tango.integrations.gs.common import (
    Constants,
    GSArtifact,
    GSArtifactConflict,
    GSArtifactNotFound,
    GSArtifactWriteError,
    GSClient,
)
from tango.step import Step
from tango.step_cache import StepCache
from tango.step_caches.remote_step_cache import RemoteNotFoundError, RemoteStepCache
from tango.step_info import StepInfo

logger = logging.getLogger(__name__)


@StepCache.register("gs")
class GSStepCache(RemoteStepCache):
    """
    This is a :class:`~tango.step_cache.StepCache` that's used by :class:`GSWorkspace`.
    It stores the results of steps on Google cloud buckets as blobs.

    It also keeps a limited in-memory cache as well as a local backup on disk, so fetching a
    step's resulting subsequent times should be fast.

    .. tip::
        Registered as a :class:`~tango.step_cache.StepCache` under the name "gs".

    :param bucket_name: The name of the google cloud bucket to use.
    :param client: The google cloud storage client to use.
    """

    Constants = Constants

    def __init__(self, bucket_name: str, client: Optional[GSClient] = None):
        if client is not None:
            assert (
                bucket_name == client.bucket_name
            ), "Assert that bucket name is same as client bucket until we do better"
            self.bucket_name = bucket_name
            self._client = client
        else:
            self._client = GSClient(bucket_name)
        super().__init__(
            tango_cache_dir() / "gs_cache" / make_safe_filename(self._client.bucket_name)
        )

    @property
    def client(self):
        return self._client

    def _step_result_remote(self, step: Union[Step, StepInfo]) -> Optional[GSArtifact]:
        """
        Returns a `GSArtifact` object containing the details of the step.
        This only returns if the step has been finalized (committed).
        """
        try:
            artifact = self.client.get(self.Constants.step_artifact_name(step))
            return artifact if artifact.committed else None
        except GSArtifactNotFound:
            return None

    def _upload_step_remote(self, step: Step, objects_dir: Path) -> GSArtifact:
        """
        Uploads the step's output to remote location.
        """
        artifact_name = self.Constants.step_artifact_name(step)
        try:
            self.client.create(artifact_name)
        except GSArtifactConflict:
            pass
        try:
            self.client.upload(artifact_name, objects_dir)
            self.client.commit(artifact_name)
        except GSArtifactWriteError:
            pass

        return self.client.get(artifact_name)

    def _download_step_remote(self, step_result, target_dir: PathOrStr) -> None:
        """
        Downloads the step's output from remote location.
        """
        try:
            self.client.download(step_result, target_dir)
        except GSArtifactNotFound:
            raise RemoteNotFoundError()

    def __len__(self):
        """
        Returns the number of committed step outputs present in the remote location.
        """
        # NOTE: lock files should not count here.
        return sum(
            1
            for ds in self.client.artifacts(
                prefix=self.Constants.STEP_ARTIFACT_PREFIX, uncommitted=False
            )
            if ds.name is not None
            and ds.name.startswith(self.Constants.STEP_ARTIFACT_PREFIX)
            and not ds.name.endswith(self.Constants.LOCK_ARTIFACT_SUFFIX)
        )
