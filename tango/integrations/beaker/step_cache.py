import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional, Union

from beaker import Beaker, Dataset, DatasetConflict, DatasetNotFound, DatasetWriteError

from tango.common.file_lock import FileLock
from tango.common.params import Params
from tango.common.util import tango_cache_dir
from tango.step import Step
from tango.step_cache import CacheMetadata, StepCache
from tango.step_caches.local_step_cache import LocalStepCache
from tango.step_info import StepInfo

from .common import Constants, step_dataset_name

logger = logging.getLogger(__name__)


@StepCache.register("beaker")
class BeakerStepCache(LocalStepCache):
    """
    This is a :class:`~tango.step_cache.StepCache` that's used by :class:`BeakerWorkspace`.
    It stores the results of steps on Beaker as datasets.

    It also keeps a limited in-memory cache as well as a local backup on disk, so fetching a
    step's resulting subsequent times should be fast.

    :param workspace: The name or ID of the Beaker workspace to use.
    :param beaker: The Beaker client to use.

    .. tip::
        Registered as :class:`~tango.step_cache.StepCache` under the name "beaker".
    """

    def __init__(self, beaker_workspace: Optional[str] = None, beaker: Optional[Beaker] = None):
        super().__init__(tango_cache_dir() / "beaker_cache")
        self.beaker: Beaker
        if beaker is not None:
            self.beaker = beaker
            if beaker_workspace is not None:
                self.beaker.config.default_workspace = beaker_workspace
                self.beaker.workspace.ensure(beaker_workspace)
        elif beaker_workspace is not None:
            self.beaker = Beaker.from_env(default_workspace=beaker_workspace)
        else:
            self.beaker = Beaker.from_env()

    def _acquire_step_lock_file(self, step: Union[Step, StepInfo], read_only_ok: bool = False):
        return FileLock(
            self.step_dir(step).with_suffix(".lock"), read_only_ok=read_only_ok
        ).acquire_with_updates(desc=f"acquiring step cache lock for '{step.unique_id}'")

    def _step_result_dataset(self, step: Union[Step, StepInfo]) -> Optional[Dataset]:
        try:
            dataset = self.beaker.dataset.get(step_dataset_name(step))
            return dataset if dataset.committed is not None else None
        except DatasetNotFound:
            return None

    def _sync_step_dataset(self, step: Step, objects_dir: Path) -> Dataset:
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

    def __contains__(self, step: Any) -> bool:
        if isinstance(step, (Step, StepInfo)):
            cacheable = step.cache_results if isinstance(step, Step) else step.cacheable
            if not cacheable:
                return False
            return self._step_result_dataset(step) is not None
        else:
            return False

    def __getitem__(self, step: Union[Step, StepInfo]) -> Any:
        key = step.unique_id
        dataset = self._step_result_dataset(step)
        if dataset is None:
            raise KeyError(step)

        # Try getting the result from our in-memory caches first.
        result = self._get_from_cache(key)
        if result is not None:
            return result

        def load_and_return():
            metadata = CacheMetadata.from_params(Params.from_file(self._metadata_path(step)))
            result = metadata.format.read(self.step_dir(step) / Constants.STEP_RESULT_DIR)
            self._add_to_cache(key, result)
            return result

        # Next check our local on-disk cache.
        with self._acquire_step_lock_file(step, read_only_ok=True):
            if self.step_dir(step).is_dir():
                return load_and_return()

        # Finally, check Beaker for the corresponding dataset.
        with self._acquire_step_lock_file(step):
            # Make sure the step wasn't cached since the last time we checked (above).
            if self.step_dir(step).is_dir():
                return load_and_return()

            # We'll download the dataset to a temporary directory first, in case something goes wrong.
            temp_dir = tempfile.mkdtemp(dir=self.dir, prefix=key)
            try:
                self.beaker.dataset.fetch(dataset, target=temp_dir, quiet=True)
                # Download and extraction was successful, rename temp directory to final step result directory.
                os.replace(temp_dir, self.step_dir(step))
            except DatasetNotFound:
                raise KeyError(step)
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

            return load_and_return()

    def __setitem__(self, step: Step, value: Any) -> None:
        if not step.cache_results:
            logger.warning("Tried to cache step %s despite being marked as uncacheable.", step.name)
            return

        with self._acquire_step_lock_file(step):
            # We'll write the step's results to temporary directory first, and try to upload to
            # Beaker from there in case anything goes wrong.
            temp_dir = Path(tempfile.mkdtemp(dir=self.dir, prefix=step.unique_id))
            (temp_dir / Constants.STEP_RESULT_DIR).mkdir()
            try:
                step.format.write(value, temp_dir / Constants.STEP_RESULT_DIR)
                metadata = CacheMetadata(step=step.unique_id, format=step.format)
                metadata.to_params().to_file(temp_dir / self.METADATA_FILE_NAME)
                # Create the dataset and upload serialized result to it.
                self._sync_step_dataset(step, temp_dir)
                # Upload successful, rename temp directory to the final step result directory.
                if self.step_dir(step).is_dir():
                    shutil.rmtree(self.step_dir(step), ignore_errors=True)
                os.replace(temp_dir, self.step_dir(step))
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

        # Finally, add to in-memory caches.
        self._add_to_cache(step.unique_id, value)

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
