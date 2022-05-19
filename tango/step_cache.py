import logging
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, TypeVar, Union

from .common.from_params import FromParams
from .common.registrable import Registrable
from .format import Format
from .step import Step
from .step_info import StepInfo

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
        if not isinstance(step, (Step, StepInfo)):
            return False
        try:
            self.__getitem__(step)
            return True
        except KeyError:
            return False

    @abstractmethod
    def __getitem__(self, step: Union[Step, StepInfo]) -> Any:
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

    format: Format
    """
    The format used to serialize the step's result.
    """
