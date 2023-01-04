import logging
import os
import shutil
import tempfile
from abc import abstractmethod
from pathlib import Path
from typing import Any, Optional, Union

from tango.common.aliases import PathOrStr
from tango.common.exceptions import TangoError
from tango.common.file_lock import FileLock
from tango.common.params import Params
from tango.common.remote_utils import (
    RemoteClient,
    RemoteConstants,
    RemoteDataset,
    RemoteDatasetConflict,
    RemoteDatasetNotFound,
    RemoteDatasetWriteError,
)
from tango.step import Step
from tango.step_cache import CacheMetadata
from tango.step_caches.local_step_cache import LocalStepCache
from tango.step_info import StepInfo

logger = logging.getLogger(__name__)


class RemoteNotFoundError(TangoError):
    """
    Classes inheriting from the RemoteStepCache should raise this if a step result object is not found.
    """


# This class inherits from `LocalStepCache` to benefit from its in-memory "weak cache" and "strong cache",
# but it handles saving artifacts to disk a little differently.
class RemoteStepCache(LocalStepCache):
    """
    This is a :class:`~tango.step_cache.StepCache` that's used by :class:`RemoteWorkspace`.
    It stores the results of steps on some RemoteWorkspace.

    It also keeps a limited in-memory cache as well as a local backup on disk, so fetching a
    step's resulting subsequent times should be fast.

    .. tip::
        All remote step caches inherit from this.
    """

    Constants = RemoteConstants

    def __init__(self, local_dir: Path):
        super().__init__(local_dir)

    @property
    @abstractmethod
    def client(self) -> RemoteClient:
        raise NotImplementedError()

    def _step_result_remote(self, step: Union[Step, StepInfo]) -> Optional[RemoteDataset]:
        """ """
        try:
            dataset = self.client.get(self.Constants.step_dataset_name(step))
            return dataset if dataset.committed else None
        except RemoteDatasetNotFound:
            return None

    def _sync_step_remote(self, step: Step, objects_dir: Path) -> RemoteDataset:
        dataset_name = self.Constants.step_dataset_name(step)
        try:
            self.client.create(dataset_name, commit=False)
        except RemoteDatasetConflict:
            pass
        try:
            self.client.sync(dataset_name, objects_dir)
            self.client.commit(dataset_name)
        except RemoteDatasetWriteError:
            pass

        return self.client.get(dataset_name)

    def _fetch_step_remote(self, step_result, target_dir: PathOrStr) -> None:
        try:
            self.client.fetch(step_result, target_dir)
        except RemoteDatasetNotFound:
            self._raise_remote_not_found()

    def __len__(self):
        # NOTE: lock datasets should not count here. They start with the same prefix,
        # but they never get committed.
        return sum(
            1
            for ds in self.client.datasets(
                uncommitted=False, match=self.Constants.STEP_DATASET_PREFIX
            )
            if ds.name is not None and ds.name.startswith(self.Constants.STEP_DATASET_PREFIX)
        )

    def _step_results_dir(self) -> str:
        return self.Constants.STEP_RESULT_DIR

    def _raise_remote_not_found(self):
        raise RemoteNotFoundError()

    def _acquire_step_lock_file(self, step: Union[Step, StepInfo], read_only_ok: bool = False):
        return FileLock(
            self.step_dir(step).with_suffix(".lock"), read_only_ok=read_only_ok
        ).acquire_with_updates(desc=f"acquiring step cache lock for '{step.unique_id}'")

    def __contains__(self, step: Any) -> bool:
        if isinstance(step, (Step, StepInfo)):
            cacheable = step.cache_results if isinstance(step, Step) else step.cacheable
            if not cacheable:
                return False

            # TODO: old beaker step cache seems to not check locally. Should it?
            key = step.unique_id

            # First check if we have a copy in memory.
            if key in self.strong_cache:
                return True
            if key in self.weak_cache:
                return True

            # Then check if we have a copy on disk in our cache directory.
            with self._acquire_step_lock_file(step, read_only_ok=True):
                if self.step_dir(step).is_dir():
                    return True

            # If not, check the remote location.
            return self._step_result_remote(step) is not None
        else:
            return False

    def __getitem__(self, step: Union[Step, StepInfo]) -> Any:
        key = step.unique_id
        step_result = self._step_result_remote(step)
        if step_result is None:
            raise KeyError(step)

        # Try getting the result from our in-memory caches first.
        result = self._get_from_cache(key)
        if result is not None:
            return result

        def load_and_return():
            metadata = CacheMetadata.from_params(Params.from_file(self._metadata_path(step)))
            result = metadata.format.read(self.step_dir(step) / self._step_results_dir())
            self._add_to_cache(key, result)
            return result

        # Next check our local on-disk cache.
        with self._acquire_step_lock_file(step, read_only_ok=True):
            if self.step_dir(step).is_dir():
                return load_and_return()

        # Finally, check the remote location for the corresponding dataset.
        with self._acquire_step_lock_file(step):
            # Make sure the step wasn't cached since the last time we checked (above).
            if self.step_dir(step).is_dir():
                return load_and_return()

            # We'll download the dataset to a temporary directory first, in case something goes wrong.
            temp_dir = tempfile.mkdtemp(dir=self.dir, prefix=key)
            try:
                self._fetch_step_remote(step_result, target_dir=temp_dir)
                # Download and extraction was successful, rename temp directory to final step result directory.
                os.replace(temp_dir, self.step_dir(step))
            except RemoteNotFoundError:
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
            # remote workspace from there in case anything goes wrong.
            temp_dir = Path(tempfile.mkdtemp(dir=self.dir, prefix=step.unique_id))
            (temp_dir / self._step_results_dir()).mkdir()
            try:
                step.format.write(value, temp_dir / self._step_results_dir())
                metadata = CacheMetadata(step=step.unique_id, format=step.format)
                metadata.to_params().to_file(temp_dir / self.METADATA_FILE_NAME)
                # Create the dataset and upload serialized result to it.
                self._sync_step_remote(step, temp_dir)
                # Upload successful, rename temp directory to the final step result directory.
                if self.step_dir(step).is_dir():
                    shutil.rmtree(self.step_dir(step), ignore_errors=True)
                os.replace(temp_dir, self.step_dir(step))
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

        # Finally, add to in-memory caches.
        self._add_to_cache(step.unique_id, value)
