import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional, Union

from tango.common.aliases import PathOrStr
from tango.common.exceptions import TangoError
from tango.common.file_lock import FileLock
from tango.common.params import Params
from tango.step import Step
from tango.step_cache import CacheMetadata
from tango.step_caches.local_step_cache import LocalStepCache
from tango.step_info import StepInfo


# TODO: move this somewhere
class Constants:
    RUN_DATASET_PREFIX = "tango-run-"
    RUN_DATA_FNAME = "run.json"
    STEP_DATASET_PREFIX = "tango-step-"
    STEP_INFO_FNAME = "step_info.json"
    STEP_RESULT_DIR = "result"
    ENTRYPOINT_DATASET_PREFIX = "tango-entrypoint-"
    STEP_GRAPH_DATASET_PREFIX = "tango-step-graph-"
    STEP_EXPERIMENT_PREFIX = "tango-step-"


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

    def __init__(self, local_dir: Path):
        super().__init__(local_dir)

    def _acquire_step_lock_file(self, step: Union[Step, StepInfo], read_only_ok: bool = False):
        return FileLock(
            self.step_dir(step).with_suffix(".lock"), read_only_ok=read_only_ok
        ).acquire_with_updates(desc=f"acquiring step cache lock for '{step.unique_id}'")

    def __contains__(self, step: Any) -> bool:
        if isinstance(step, (Step, StepInfo)):
            cacheable = step.cache_results if isinstance(step, Step) else step.cacheable
            if not cacheable:
                return False

            # TODO: beaker seems to not check locally. Should it?
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

    # TODO: change output type
    def _step_result_remote(self, step: Union[Step, StepInfo]) -> Optional[Any]:
        raise NotImplementedError()

    def _fetch_step_remote(self, step_result, target_dir: PathOrStr):
        raise NotImplementedError()

    def _raise_remote_not_found(self):
        raise RemoteNotFoundError()

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
            # TODO: note that this is slightly different for beaker and wandb
            result = metadata.format.read(self.step_dir(step) / Constants.STEP_RESULT_DIR)
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

    # TODO: change output type
    def _sync_step_remote(self, step: Step, objects_dir: Path) -> Any:
        raise NotImplementedError()

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
                self._sync_step_remote(step, temp_dir)
                # Upload successful, rename temp directory to the final step result directory.
                if self.step_dir(step).is_dir():
                    shutil.rmtree(self.step_dir(step), ignore_errors=True)
                os.replace(temp_dir, self.step_dir(step))
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

        # Finally, add to in-memory caches.
        self._add_to_cache(step.unique_id, value)

    def __len__(self):
        raise NotImplementedError()