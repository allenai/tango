import logging
from typing import Optional

from tango.common.util import make_safe_filename, tango_cache_dir
from tango.step_cache import StepCache
from tango.step_caches.remote_step_cache import RemoteStepCache

from .common import Constants, GCSClient

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

    Constants = Constants

    def __init__(self, bucket_name: str, client: Optional[GCSClient] = None):
        if client is not None:
            assert (
                bucket_name == client.bucket_name
            ), "Assert that bucket name is same as client bucket until we do better"
            self.bucket_name = bucket_name
            self._client = client
        else:
            self._client = GCSClient(bucket_name)
        super().__init__(
            tango_cache_dir() / "gs_cache" / make_safe_filename(self._client.bucket_name)
        )

    @property
    def client(self):
        return self._client
