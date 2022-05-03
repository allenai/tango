import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional, Union

import wandb
from wandb.errors import Error as WandbError

from tango.common.aliases import PathOrStr
from tango.common.file_lock import FileLock
from tango.common.params import Params
from tango.common.util import tango_cache_dir
from tango.step import Step
from tango.step_cache import CacheMetadata, StepCache
from tango.step_caches.local_step_cache import LocalStepCache
from tango.step_info import StepInfo

from .util import ArtifactKind, check_environment, is_missing_artifact_error

logger = logging.getLogger(__name__)


# This class inherits from `LocalStepCache` to benefit from its in-memory "weak cache" and "strong cache",
# but it handles saving artifacts to disk a little differently.
@StepCache.register("wandb")
class WandbStepCache(LocalStepCache):
    """
    This is a :class:`~tango.step_cache.StepCache` that's used by :class:`WandbWorkspace`.
    It stores the results of steps on W&B as Artifacts.

    It also keeps a limited in-memory cache as well as a local backup on disk, so fetching a
    step's resulting subsequent times should be fast.

    :param project: The W&B project to use.
    :param entity: The W&B entity (user or organization account) to use.

    .. tip::
        Registered as :class:`~tango.step_cache.StepCache` under the name "wandb".
    """

    def __init__(self, project: str, entity: str):
        check_environment()
        super().__init__(tango_cache_dir() / "wandb_cache")
        self.project = project
        self.entity = entity

    @property
    def wandb_client(self) -> wandb.Api:
        return wandb.Api(overrides={"entity": self.entity, "project": self.project})

    def _acquire_step_lock_file(self, step: Union[Step, StepInfo], read_only_ok: bool = False):
        return FileLock(
            self.step_dir(step).with_suffix(".lock"), read_only_ok=read_only_ok
        ).acquire_with_updates(desc=f"acquiring step cache lock for '{step.unique_id}'")

    def _step_artifact_name(self, step: Union[Step, StepInfo]) -> str:
        if isinstance(step, Step):
            return step.__class__.__name__
        else:
            return step.step_class_name

    def get_step_result_artifact(
        self, step: Union[Step, StepInfo]
    ) -> Optional[wandb.apis.public.Artifact]:
        try:
            return self.wandb_client.artifact(
                f"{self.entity}/{self.project}/{self._step_artifact_name(step)}:{step.unique_id}",
                type=ArtifactKind.STEP_RESULT.value,
            )
        except WandbError as exc:
            if is_missing_artifact_error(exc):
                return None
            else:
                raise

    def create_step_result_artifact(
        self, step: Step, objects_dir: Optional[PathOrStr] = None
    ) -> None:
        """
        Create an artifact for the result of a step.
        """
        artifact = wandb.Artifact(
            self._step_artifact_name(step), type=ArtifactKind.STEP_RESULT.value
        )

        # Add files
        if objects_dir is not None:
            artifact.add_dir(str(objects_dir))

        # Log/persist the artifact to W&B.
        artifact.save()
        artifact.wait()

        # Add an alias for the step's unique ID.
        # Only after we've logged the artifact can we add an alias.
        artifact.aliases.append(step.unique_id)
        artifact.save()
        artifact.wait()

    def use_step_result_artifact(self, step: Union[Step, StepInfo]) -> None:
        """
        "Use" the artifact corresponding to the result of a step.
        """
        if wandb.run is None:
            raise RuntimeError("This can only be called from within a W&B run")
        wandb.run.use_artifact(
            f"{self.entity}/{self.project}/{self._step_artifact_name(step)}:{step.unique_id}"
        )

    def __contains__(self, step: Any) -> bool:
        if isinstance(step, (Step, StepInfo)):
            cacheable = step.cache_results if isinstance(step, Step) else step.cacheable
            if not cacheable:
                return False

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

            # If not, check W&B for the corresponding artifact.
            return self.get_step_result_artifact(step) is not None
        else:
            return False

    def __getitem__(self, step: Union[Step, StepInfo]) -> Any:
        key = step.unique_id

        if wandb.run is not None:
            # Mark that the current run uses the step's result artifact.
            try:
                self.use_step_result_artifact(step)
            except WandbError as exc:
                if is_missing_artifact_error(exc):
                    raise KeyError(step)
                else:
                    raise

        # Try getting the result from our in-memory caches first.
        result = self._get_from_cache(key)
        if result is not None:
            return result

        def load_and_return():
            metadata = CacheMetadata.from_params(Params.from_file(self._metadata_path(step)))
            result = metadata.format.read(self.step_dir(step))
            self._add_to_cache(key, result)
            return result

        # Next check our local on-disk cache.
        with self._acquire_step_lock_file(step, read_only_ok=True):
            if self.step_dir(step).is_dir():
                return load_and_return()

        # Finally, check W&B for the corresponding artifact.
        with self._acquire_step_lock_file(step):
            # Make sure the step wasn't cached since the last time we checked (above).
            if self.step_dir(step).is_dir():
                return load_and_return()

            artifact = self.get_step_result_artifact(step)
            if artifact is None:
                raise KeyError(step)

            # We'll download the artifact to a temporary directory first, in case something goes wrong.
            temp_dir = tempfile.mkdtemp(dir=self.dir, prefix=key)
            try:
                artifact.download(root=temp_dir, recursive=True)
                # Download and extraction was successful, rename temp directory to final step result directory.
                os.replace(temp_dir, self.step_dir(step))
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

            return load_and_return()

    def __setitem__(self, step: Step, value: Any) -> None:
        if not step.cache_results:
            logger.warning("Tried to cache step %s despite being marked as uncacheable.", step.name)
            return

        if wandb.run is None:
            raise RuntimeError("Can only add results to the WandbStepCache within a W&B run")

        with self._acquire_step_lock_file(step):
            # We'll write the step's results to temporary directory first, and try to upload to W&B
            # from there in case anything goes wrong.
            temp_dir = Path(tempfile.mkdtemp(dir=self.dir, prefix=step.unique_id))
            try:
                step.format.write(value, temp_dir)
                metadata = CacheMetadata(step=step.unique_id, format=step.format)
                metadata.to_params().to_file(temp_dir / self.METADATA_FILE_NAME)
                # Create the artifact and upload serialized result to it.
                self.create_step_result_artifact(step, temp_dir)
                # Upload successful, rename temp directory to the final step result directory.
                if self.step_dir(step).is_dir():
                    shutil.rmtree(self.step_dir(step), ignore_errors=True)
                os.replace(temp_dir, self.step_dir(step))
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

        # Finally, add to in-memory caches.
        self._add_to_cache(step.unique_id, value)

    def __len__(self) -> int:
        completed_cacheable_step_runs = self.wandb_client.runs(
            f"{self.entity}/{self.project}",
            filters={
                "config.job_type": "step",
                "config.cacheable": True,
                "state": "finished",
            },
        )
        return len(list(completed_cacheable_step_runs))
