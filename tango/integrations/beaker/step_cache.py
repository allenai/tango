import logging
from pathlib import Path
from typing import Optional, Union

from beaker import Beaker
from beaker import Dataset as BeakerDataset
from beaker import DatasetConflict, DatasetNotFound, DatasetWriteError

from tango import Step
from tango.common import PathOrStr
from tango.common.exceptions import ConfigurationError
from tango.common.util import make_safe_filename, tango_cache_dir
from tango.integrations.beaker.common import Constants, get_client
from tango.step_cache import StepCache
from tango.step_caches.remote_step_cache import RemoteNotFoundError, RemoteStepCache
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

    def _step_result_remote(self, step: Union[Step, StepInfo]) -> Optional[BeakerDataset]:
        """
        Returns a `BeakerDataset` object containing the details of the step.
        This only returns if the step has been finalized (committed).
        """
        try:
            dataset = self.beaker.dataset.get(self.Constants.step_artifact_name(step))
            return dataset if dataset.committed is not None else None
        except DatasetNotFound:
            return None

    def _upload_step_remote(self, step: Step, objects_dir: Path) -> BeakerDataset:
        """
        Uploads the step's output to remote location.
        """
        dataset_name = self.Constants.step_artifact_name(step)
        try:
            self.beaker.dataset.create(dataset_name, commit=False)
        except DatasetConflict:
            pass
        try:
            self.beaker.dataset.sync(dataset_name, objects_dir, quiet=True)
            self.beaker.dataset.commit(dataset_name)
        except DatasetWriteError:
            pass

        return self.beaker.dataset.get(dataset_name)

    def _download_step_remote(self, step_result, target_dir: PathOrStr) -> None:
        """
        Downloads the step's output from remote location.
        """
        try:
            self.beaker.dataset.fetch(step_result, target_dir, quiet=True)
        except DatasetNotFound:
            raise RemoteNotFoundError()

    def __len__(self):
        """
        Returns the number of committed step outputs present in the remote location.
        """
        # NOTE: lock datasets should not count here.
        return sum(
            1
            for ds in self.beaker.workspace.iter_datasets(
                match=self.Constants.STEP_ARTIFACT_PREFIX, uncommitted=False, results=False
            )
            if ds.name is not None
            and ds.name.startswith(self.Constants.STEP_ARTIFACT_PREFIX)
            and not ds.name.endswith(self.Constants.LOCK_ARTIFACT_SUFFIX)
        )
