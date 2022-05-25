from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, Optional, Set

import pytz

from .common.util import local_timezone
from .step import Step


class StepState(Enum):
    """Describes the possible state a step can be in."""

    INCOMPLETE = "incomplete"
    """The step has not run yet."""

    RUNNING = "running"
    """The step is running right now."""

    COMPLETED = "completed"
    """The step finished running successfully."""

    FAILED = "failed"
    """The step ran, but failed."""

    UNCACHEABLE = "uncacheable"
    """The step is uncacheable. It will be executed as many times as the results are needed,
    so we don't keep track of the state."""


@dataclass
class StepInfo:
    """Stores step information without being the :class:`.Step` itself.

    It's not always possible to get a :class:`.Step` object, because :class:`.Step` objects can't be serialized.
    But you can always serialize a :class:`.StepInfo` object.
    """

    unique_id: str
    """
    The unique ID of the step
    """

    step_class_name: str
    """
    The name of the :class:`.Step` class
    """

    dependencies: Set[str]
    """
    The unique ids of all the steps that this step depends on
    """

    cacheable: bool
    """
    Whether or not the step is cacheable.
    """

    step_name: Optional[str] = None
    """
    The name of the step, if it has one. Anonymous steps are identified only by their unique ID.

    The same step can have different names in different runs. The last run wins, so don't rely
    on this property in your code. It is just here to aid readability.
    """

    version: Optional[str] = None
    """
    The version string of the :class:`.Step`, if it has one.
    """

    start_time: Optional[datetime] = None
    """
    The time (in UTC) that this step started running.

    .. seealso::
        :meth:`start_time_local()`.
    """

    end_time: Optional[datetime] = None
    """
    The time (in UTC) that this step stopped running. This will be set whether the step succeeded or failed.

    .. seealso::
        :meth:`end_time_local()`.
    """

    error: Optional[str] = None
    """
    If the step failed, this is where the error goes.

    .. note::
        Some ``Workspace`` implementations need to serialize ``StepInfo`` (using pickle or dill, for example),
        but some exceptions can't be pickled. In those cases ``error`` will just be a string representation
        of the exception.
    """

    result_location: Optional[str] = None
    """
    Location of the result. This could be a path or a URL.
    """

    @property
    def start_time_local(self) -> Optional[datetime]:
        """
        The time the step started running with respect to the local timezone, if the timezone
        can be determined.
        """
        return None if self.start_time is None else self.start_time.astimezone(local_timezone())

    @property
    def end_time_local(self) -> Optional[datetime]:
        """
        The time the step stopped running with respect to the local timezone, if the timezone
        can be determined.
        """
        return None if self.end_time is None else self.end_time.astimezone(local_timezone())

    @property
    def duration(self) -> Optional[timedelta]:
        """
        The time it took to run this step.
        """
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        else:
            return None

    @property
    def state(self) -> StepState:
        """
        Returns the state of the step
        """
        if self.cacheable:
            if self.start_time is None and self.end_time is None and self.error is None:
                return StepState.INCOMPLETE
            if self.start_time is not None and self.end_time is None and self.error is None:
                return StepState.RUNNING
            if self.start_time is not None and self.end_time is not None and self.error is None:
                return StepState.COMPLETED
            if self.start_time is not None and self.end_time is not None and self.error is not None:
                return StepState.FAILED
        else:
            return StepState.UNCACHEABLE
        raise RuntimeError(f"{self.__class__.__name__} is in an invalid state.")

    def to_json_dict(self) -> Dict[str, Any]:
        """
        Generates a JSON-safe, human-readable, dictionary representation of this dataclass.
        """
        return {
            k: (
                v.strftime("%Y-%m-%dT%H:%M:%S")
                if isinstance(v, datetime)
                else list(v)
                if isinstance(v, set)
                else v
            )
            for k, v in asdict(self).items()
        }

    @classmethod
    def from_json_dict(cls, json_dict: Dict[str, Any]) -> "StepInfo":
        """
        The inverse of :meth:`to_json_dict()`.

        :param json_dict: A dictionary representation, such as the one produced by :meth:`to_json_dict()`.
        """
        return cls(
            **{
                k: (
                    datetime.strptime(v, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=pytz.utc)
                    if k in {"start_time", "end_time"} and v is not None
                    else set(v)
                    if k == "dependencies"
                    else v
                )
                for k, v in json_dict.items()
            }
        )

    @classmethod
    def new_from_step(cls, step: Step, **kwargs) -> "StepInfo":
        return cls(
            unique_id=step.unique_id,
            step_name=step.name,
            step_class_name=step.__class__.__name__,
            version=step.VERSION,
            dependencies={dep.unique_id for dep in step.dependencies},
            cacheable=step.cache_results,
            **kwargs,
        )
