import logging
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, TypeVar

from tango.common.from_params import FromParams
from tango.common.registrable import Registrable
from tango.step import Step

logger = logging.getLogger(__name__)


T = TypeVar("T")


class StepCache(Registrable):
    """
    This is a mapping from instances of :class:`~tango.step.Step` to the results of that step.
    Generally :class:`StepCache` implementations are used internally by :class:`~tango.workspace.Workspace`
    implementations.
    """

    default_implementation = "memory"
    """
    The default implementation is :class:`.MemoryStepCache`.
    """

    def __contains__(self, step: Any) -> bool:
        """This is a generic implementation of ``__contains__``. If you are writing your own
        ``StepCache``, you might want to write a faster one yourself."""
        if not isinstance(step, Step):
            return False
        try:
            self.__getitem__(step)
            return True
        except KeyError:
            return False

    @abstractmethod
    def __getitem__(self, step: Step) -> Any:
        """Returns the results for the given step."""
        raise NotImplementedError()

    @abstractmethod
    def __setitem__(self, step: Step, value: Any) -> None:
        """Writes the results for the given step. Throws an exception if the step is already cached."""
        raise NotImplementedError()

    @abstractmethod
    def __len__(self) -> int:
        """Returns the number of results saved in this cache."""
        raise NotImplementedError()


@dataclass
class CacheMetadata(FromParams):
    step: str
    """
    The step name.
    """
