import logging
from typing import Any, Dict

from tango.step import Step
from tango.step_cache import StepCache

logger = logging.getLogger(__name__)


@StepCache.register("memory")
class MemoryStepCache(StepCache):
    """
    This is a :class:`.StepCache` that stores results in memory. It is little more than a Python dictionary.

    .. tip::
        Registered as :class:`.StepCache` under the name "memory".
    """

    def __init__(self):
        self.cache: Dict[str, Any] = {}

    def __getitem__(self, step: Step) -> Any:
        return self.cache[step.unique_id]

    def __setitem__(self, step: Step, value: Any) -> None:
        if step in self:
            raise ValueError(f"{step.unique_id} is already cached! Will not overwrite.")
        if step.cache_results:
            self.cache[step.unique_id] = value
        else:
            logger.warning("Tried to cache step %s despite being marked as uncacheable.", step.name)

    def __contains__(self, step: object):
        if isinstance(step, Step):
            return step.unique_id in self.cache
        else:
            return False

    def __len__(self) -> int:
        return len(self.cache)


default_step_cache = MemoryStepCache()
