import logging
from typing import Optional, Union

from beaker import Beaker

from tango.common.exceptions import ConfigurationError
from tango.common.remote_utils import RemoteDatasetNotFound
from tango.common.util import make_safe_filename, tango_cache_dir
from tango.integrations.beaker.common import Constants, get_client
from tango.step import Step
from tango.step_cache import StepCache
from tango.step_caches.remote_step_cache import RemoteStepCache
from tango.step_info import StepInfo

logger = logging.getLogger(__name__)


@StepCache.register("beaker")
class BeakerStepCache(RemoteStepCache):
    """
    This is a :class:`~tango.step_cache.StepCache` that's used by :class:`BeakerWorkspace`.
    It stores the results of steps on Beaker as datasets.

    It also keeps a limited in-memory cache as well as a local backup on disk, so fetching a
    step's resulting subsequent times should be fast.

    .. tip::
        Registered as a :class:`~tango.step_cache.StepCache` under the name "beaker".

    :param workspace: The name or ID of the Beaker workspace to use.
    :param beaker: The Beaker client to use.
    """

    Constants = Constants

    def __init__(self, beaker_workspace: Optional[str] = None, beaker: Optional[Beaker] = None):
        self._client = get_client(beaker_workspace=beaker_workspace, beaker=beaker)
        if self._client.beaker.config.default_workspace is None:
            raise ConfigurationError("Beaker default workspace must be set")
        super().__init__(
            tango_cache_dir()
            / "beaker_cache"
            / make_safe_filename(self._client.beaker.config.default_workspace)
        )

    @property
    def client(self):
        return self._client

    def _step_result_remote(self, step: Union[Step, StepInfo]):
        try:
            dataset = self.client.get(self.Constants.step_dataset_name(step))
            return dataset if dataset.committed else None
        except RemoteDatasetNotFound:
            return None
