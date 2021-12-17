from abc import abstractmethod
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Iterable, Iterator, List, Optional, Set, TypeVar, Union

import dill
import petname

from tango import step_cache
from tango.common import Registrable
from tango.step import Step
from tango.step_cache import StepCache

T = TypeVar("T")


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

    step_name: Optional[str]
    """
    The name of the step, if it has one. Anonymous steps are identified only by their unique ID.
    """

    step_class_name: str
    """
    The name of the :class:`.Step` class
    """

    version: Optional[str]
    """
    The version string of the :class:`.Step`, if it has one
    """

    dependencies: Set[str]
    """
    The unique ids of all the steps that this step depends on
    """

    start_time: Optional[datetime] = None
    """
    The time this step started running
    """

    end_time: Optional[datetime] = None
    """
    The time this step stopped running. This will be set whether the step succeeded or failed.
    """

    error: Optional[Union[BaseException, str]] = None
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
        if self.start_time is None and self.end_time is None and self.error is None:
            return StepState.INCOMPLETE
        if self.start_time is not None and self.end_time is None and self.error is None:
            return StepState.RUNNING
        if self.start_time is not None and self.end_time is not None and self.error is None:
            return StepState.COMPLETED
        if self.start_time is not None and self.end_time is not None and self.error is not None:
            return StepState.FAILED
        raise RuntimeError(f"{self.__class__.__name__} is in an invalid state.")

    def serialize(self) -> bytes:
        """
        Returns a serialized form of the ``StepInfo``.
        """
        instance_to_dump = self
        if isinstance(self.error, BaseException):
            # See if we can pickle and unpickle the exception.
            # When we can't, we'll fallback to storing the exception as a string
            # representation of it.
            dump = dill.dumps(self.error)
            try:
                dill.loads(dump)
            except TypeError:
                # Fails with TypeError for some exceptions that take multiple positional
                # arguments.
                instance_to_dump = replace(self, error=repr(self.error))
        return dill.dumps(instance_to_dump)

    @classmethod
    def deserialize(cls, data: bytes) -> "StepInfo":
        """
        Deserialize the result of :meth:`serialize()` into a ``StepInfo`` instance.
        """
        return dill.loads(data)


class Workspace(Registrable):
    """
    A workspace is a place for Tango to put the results of steps, intermediate results, and various other pieces
    of metadata. If you don't want to worry about all that, do nothing and Tango will use the default
    :class:`.LocalWorkspace` that puts everything into a directory of your choosing.

    If you want to do fancy things like store results in the cloud, share state across machines, etc., this is your
    integration point.

    If you got here solely because you want to share results between machines, consider that
    :class:`.LocalWorkspace` works fine on an NFS drive.
    """

    default_implementation = "local"

    #
    # As a general rule, workspaces can never return `Step`, only `StepInfo`, because we can't reliably serialize
    # objects of type `Step`. Doing that would require serializing the code that runs the step, and we can't
    # do that.
    #

    @property
    @abstractmethod
    def step_cache(self) -> StepCache:
        """
        A :class:`.StepCache` to store step results in
        """
        raise NotImplementedError()

    def work_dir(self, step: Step) -> Path:
        """Steps that can be restarted (like a training job that gets interrupted half-way through)
        must save their state somewhere. A :class:`.StepCache` can help by providing a suitable location
        in this method.

        By default, the step dir is a temporary directory that gets cleaned up after every run.
        This effectively disables restartability of steps."""

        # TemporaryDirectory cleans up the directory automatically when the process exits. Neat!
        return Path(TemporaryDirectory(prefix=f"{step.unique_id}-", suffix=".step_dir").name)

    @abstractmethod
    def step_info(self, step_or_unique_id: Union[Step, str]) -> StepInfo:
        """
        Returns a :class:`.StepInfo` for a given step
        """
        raise NotImplementedError()

    @abstractmethod
    def step_starting(self, step: Step) -> None:
        """
        The :class:`.Step` class calls this when a step is about to start running.

        :param step: The step that is about to start.
        """
        raise NotImplementedError()

    @abstractmethod
    def step_finished(self, step: Step, result: T) -> T:
        """
        The :class:`.Step` class calls this when a step finished running.

        :param step: The step that finished.

        This method is given the result of the step's :meth:`.Step.run` method. It is expected to return that
        result. This gives it the opportunity to make changes to the result if necessary. For example, if the
        :meth:`.Step.run` method returns an iterator, that iterator would be consumed when it's written to the
        cache. So this method can handle the situation and return something other than the now-consumed iterator.
        """
        raise NotImplementedError()

    @abstractmethod
    def step_failed(self, step: Step, e: BaseException) -> None:
        """
        The :class:`.Step` class calls this when a step failed.

        :param step: The step that failed.
        :param e: The exception thrown by the step's :meth:`.Step.run` method.
        """
        raise NotImplementedError()

    @abstractmethod
    def register_run(self, targets: Iterable[Step], name: Optional[str] = None) -> str:
        """
        Register a run in the workspace. A run is a set of target steps that a user wants to execute.

        :param targets: The steps that the user wants to execute. This could come from a :class:`.StepGraph`.
        :param name: A name for the run. Runs must have unique names. If not given, this method invents a name and
                     returns it.
        :return: The name for the run
        """
        raise NotImplementedError()

    @abstractmethod
    def registered_runs(self) -> List[str]:
        """
        Returns all runs in the workspace

        :return: A list of run names that are registered in the workspace
        """
        raise NotImplementedError()

    @abstractmethod
    def registered_run(self, name: str) -> Dict[str, StepInfo]:
        """
        Returns the run with the given name

        :return: A run, represented as a mapping from step name to :class:`.StepInfo`.

        Note that this dictionary only contains the targets of a run. Usually, that means it
        contains all named steps. Un-named dependencies (or dependencies that are not targets)
        are not contained in the result.

        This method throws ``KeyError`` if there is no run with the given name.
        """
        raise NotImplementedError()


@Workspace.register("memory")
class MemoryWorkspace(Workspace):
    """
    This is a workspace that keeps all its data in memory. This is useful for debugging or for quick jobs, but of
    course you don't get any caching across restarts.
    """

    def __init__(self):
        self.unique_id_to_info: Dict[str, StepInfo] = {}
        self.runs: Dict[str, Set[Step]] = {}

    @property
    def step_cache(self) -> StepCache:
        return step_cache.default_step_cache

    def step_info(self, step_or_unique_id: Union[Step, str]) -> StepInfo:
        unique_id = (
            step_or_unique_id.unique_id
            if isinstance(step_or_unique_id, Step)
            else step_or_unique_id
        )
        try:
            return self.unique_id_to_info[unique_id]
        except KeyError:
            if isinstance(step_or_unique_id, Step):
                step = step_or_unique_id
                return StepInfo(
                    step.unique_id,
                    step.name if step.name != step.unique_id else None,
                    step.__class__.__name__,
                    step.VERSION,
                    {dep.unique_id for dep in step.dependencies},
                )
            else:
                raise KeyError()

    def step_starting(self, step: Step) -> None:
        self.unique_id_to_info[step.unique_id] = StepInfo(
            step.unique_id,
            step.name if step.name != step.unique_id else None,
            step.__class__.__name__,
            step.VERSION,
            {dep.unique_id for dep in step.dependencies},
            datetime.now(),
        )

    def step_finished(self, step: Step, result: T) -> T:
        existing_step_info = self.unique_id_to_info[step.unique_id]
        if existing_step_info.state != StepState.RUNNING:
            raise RuntimeError(f"Step {step.name} is ending, but it never started.")
        existing_step_info.end_time = datetime.now()

        if step.cache_results:
            self.step_cache[step] = result
            if hasattr(result, "__next__"):
                assert isinstance(result, Iterator)
                # Caching the iterator will consume it, so we write it to the cache and then read from the cache
                # for the return value.
                return self.step_cache[step]
        return result

    def step_failed(self, step: Step, e: BaseException) -> None:
        assert e is not None
        existing_step_info = self.unique_id_to_info[step.unique_id]
        if existing_step_info.state != StepState.RUNNING:
            raise RuntimeError(f"Step {step.name} is failing, but it never started.")
        existing_step_info.end_time = datetime.now()
        existing_step_info.error = e

    def register_run(self, targets: Iterable[Step], name: Optional[str] = None) -> str:
        if name is None:
            name = petname.generate()
        self.runs[name] = set(targets)
        for step in self.runs[name]:
            self.unique_id_to_info[step.unique_id] = StepInfo(
                step.unique_id,
                step.name if step.name != step.unique_id else None,
                step.__class__.__name__,
                step.VERSION,
                {dep.unique_id for dep in step.dependencies},
            )
        return name

    def registered_runs(self) -> List[str]:
        return list(self.runs.keys())

    def registered_run(self, name: str) -> Dict[str, StepInfo]:
        return {step.unique_id: self.unique_id_to_info[step.unique_id] for step in self.runs[name]}


default_workspace = MemoryWorkspace()
