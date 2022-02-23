import collections
import logging
import weakref
from pathlib import Path
from typing import Any, MutableMapping, Optional, OrderedDict, Union

from tango.common.aliases import PathOrStr
from tango.step import Step
from tango.step_cache import CacheMetadata, StepCache

logger = logging.getLogger(__name__)


@StepCache.register("local")
class LocalStepCache(StepCache):
    """
    This is a :class:`.StepCache` that stores its results on disk, in the location given in ``dir``.

    Every cached step gets a directory under ``dir`` with that step's :attr:`~tango.step.Step.unique_id`.
    In that directory we store the results themselves in some format according to the step's
    :attr:`~tango.step.Step.FORMAT`, and we also write a ``cache-metadata.json`` file that
    stores the :class:`.CacheMetadata`.

    The presence of ``cache-metadata.json`` signifies that the cache entry is complete and
    has been written successfully.

    .. tip::
        Registered as :class:`.StepCache` under the name "local".

    """

    LRU_CACHE_MAX_SIZE = 8

    def __init__(self, dir: PathOrStr):
        self.dir = Path(dir)
        self.dir.mkdir(parents=True, exist_ok=True)

        # We keep an in-memory cache as well so we don't have to de-serialize stuff
        # we happen to have in memory already.
        self.weak_cache: MutableMapping[str, Any] = weakref.WeakValueDictionary()

        # Not all Python objects can be referenced weakly, and even if they can they
        # might get removed too quickly, so we also keep an LRU cache.
        self.strong_cache: OrderedDict[str, Any] = collections.OrderedDict()

    def _add_to_cache(self, key: str, o: Any) -> None:
        if hasattr(o, "__next__"):
            # We never cache iterators, because they are mutable, storing their current position.
            return

        self.strong_cache[key] = o
        self.strong_cache.move_to_end(key)
        while len(self.strong_cache) > self.LRU_CACHE_MAX_SIZE:
            del self.strong_cache[next(iter(self.strong_cache))]

        try:
            self.weak_cache[key] = o
        except TypeError:
            pass  # Many native Python objects cannot be referenced weakly, and they throw TypeError when you try

    def _get_from_cache(self, key: str) -> Optional[Any]:
        result = self.strong_cache.get(key)
        if result is not None:
            self.strong_cache.move_to_end(key)
            return result
        try:
            return self.weak_cache[key]
        except KeyError:
            return None

    def __contains__(self, step: object) -> bool:
        if isinstance(step, Step) and step.cache_results:
            key = step.unique_id
            if key in self.strong_cache:
                return True
            if key in self.weak_cache:
                return True
            metadata_file = self.step_dir(step) / "cache-metadata.json"
            return metadata_file.exists()
        else:
            return False

    def __getitem__(self, step: Step) -> Any:
        key = step.unique_id
        result = self._get_from_cache(key)
        if result is None:
            if step not in self:
                raise KeyError(step)
            result = step.format.read(self.step_dir(step))
            self._add_to_cache(key, result)
        return result

    def __setitem__(self, step: Step, value: Any) -> None:
        if not step.cache_results:
            logger.warning("Tried to cache step %s despite being marked as uncacheable.", step.name)
            return

        location = self.step_dir(step)
        location.mkdir(parents=True, exist_ok=True)

        metadata_location = location / "cache-metadata.json"
        if metadata_location.exists():
            raise ValueError(f"{metadata_location} already exists! Will not overwrite.")
        temp_metadata_location = metadata_location.with_suffix(".temp")

        try:
            step.format.write(value, location)
            metadata = CacheMetadata(step=step.unique_id)
            metadata.to_params().to_file(temp_metadata_location)
            self._add_to_cache(step.unique_id, value)
            temp_metadata_location.rename(metadata_location)
        except:  # noqa: E722
            try:
                temp_metadata_location.unlink()
            except FileNotFoundError:
                pass
            raise

    def __len__(self) -> int:
        return sum(1 for _ in self.dir.glob("*/cache-metadata.json"))

    def step_dir(self, step_or_unique_id: Union[Step, str]) -> Path:
        """Returns the directory that contains the results of the step.

        You can use this even for a step that's not cached yet. In that case it will return the directory where
        the results will be written."""
        if isinstance(step_or_unique_id, Step):
            if not step_or_unique_id.cache_results:
                raise RuntimeError(
                    f"Uncacheable steps (like '{step_or_unique_id.name}') don't have step directories."
                )
            unique_id = step_or_unique_id.unique_id
        else:
            unique_id = step_or_unique_id
        return self.dir / unique_id
