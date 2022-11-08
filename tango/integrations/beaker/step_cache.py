import logging
from pathlib import Path
from typing import Optional, Union

from beaker import Beaker, Dataset, DatasetConflict, DatasetNotFound, DatasetWriteError

from tango.common.aliases import PathOrStr
from tango.common.exceptions import ConfigurationError
from tango.common.util import make_safe_filename, tango_cache_dir
from tango.step import Step
from tango.step_cache import StepCache
from tango.step_caches.remote_step_cache import RemoteStepCache
from tango.step_info import StepInfo

from .common import Constants, get_client, step_dataset_name

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

    def __init__(self, beaker_workspace: Optional[str] = None, beaker: Optional[Beaker] = None):
        self.beaker: Beaker
        if beaker is not None:
            self.beaker = beaker
            if beaker_workspace is not None:
                self.beaker.config.default_workspace = beaker_workspace
                self.beaker.workspace.ensure(beaker_workspace)
        else:
            self.beaker = get_client(beaker_workspace=beaker_workspace)
        if self.beaker.config.default_workspace is None:
            raise ConfigurationError("Beaker default workspace must be set")
        super().__init__(
            tango_cache_dir()
            / "beaker_cache"
            / make_safe_filename(self.beaker.config.default_workspace)
        )

    def _step_result_remote(self, step: Union[Step, StepInfo]) -> Optional[Dataset]:
        try:
            dataset = self.beaker.dataset.get(step_dataset_name(step))
            return dataset if dataset.committed is not None else None
        except DatasetNotFound:
            return None

    def _sync_step_remote(self, step: Step, objects_dir: Path) -> Dataset:
        dataset_name = step_dataset_name(step)
        try:
            dataset = self.beaker.dataset.create(dataset_name, commit=False)
        except DatasetConflict:
            dataset = self.beaker.dataset.get(dataset_name)

        try:
            self.beaker.dataset.sync(dataset, objects_dir, quiet=True)
            dataset = self.beaker.dataset.commit(dataset)
        except DatasetWriteError:
            pass

        return dataset

    def _fetch_step_remote(self, step_result, target_dir: PathOrStr):
        try:
            self.beaker.dataset.fetch(step_result, target=target_dir, quiet=True)
        except DatasetNotFound:
            self._raise_remote_not_found()

    def __len__(self) -> int:
        # NOTE: lock datasets should not count here. They start with the same prefix,
        # but they never get committed.
        return sum(
            1
            for ds in self.beaker.workspace.datasets(
                uncommitted=False, match=Constants.STEP_DATASET_PREFIX
            )
            if ds.name is not None and ds.name.startswith(Constants.STEP_DATASET_PREFIX)
        )
