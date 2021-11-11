from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Dict, Set, TypeVar, Iterator, Iterable

import petname

import step_cache
from tango.step_cache import StepCache
from tango.step import Step
from tango.common import Registrable

T = TypeVar("T")


@dataclass
class StepInfo:
    unique_id: str
    step_name: Optional[str]
    step_class_name: str
    version: Optional[str]
    dependencies: Set[str]
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error: Optional[Exception] = None
    result_location: Optional[str] = None

    @property
    def duration(self) -> Optional[timedelta]:
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        else:
            return None

    @property
    def status(self) -> str:
        if self.start_time is None and self.end_time is None and self.error is None:
            return "incomplete"
        if self.start_time is not None and self.end_time is None and self.error is None:
            return "running"
        if self.start_time is not None and self.end_time is not None and self.error is None:
            return "completed"
        if self.start_time is not None and self.end_time is not None and self.error is not None:
            return "failed"
        raise RuntimeError(f"{self.__class__.__name__} is in an invalid state.")


class Workspace(Registrable):
    default_implementation = "local"

    #
    # As a general rule, workspaces can never return `Step`, only `StepInfo`, because we can't reliably serialize
    # objects of type `Step`. Doing that would require serializing the code that runs the step, and we can't
    # do that.
    #

    @abstractmethod
    @property
    def step_cache(self) -> StepCache:
        raise NotImplementedError()

    def work_dir(self, step: Step) -> Path:
        """Steps that can be restarted (like a training job that gets interrupted half-way through)
        must save their state somewhere. A :class:`StepCache` can help by providing a suitable location
        in this method.

        By default, the step dir is a temporary directory that gets cleaned up after every run.
        This effectively disables restartability of steps."""

        # TemporaryDirectory cleans up the directory automatically when the process exits. Neat!
        return Path(TemporaryDirectory(prefix=f"{step.unique_id}-", suffix=".step_dir").name)

    @abstractmethod
    def step_info(self, step: Step) -> StepInfo:
        raise NotImplementedError()

    def steps(self, include_completed: bool = True) -> Iterable[StepInfo]:
        raise NotImplementedError()

    @abstractmethod
    def step_started(self, step: Step) -> None:
        raise NotImplementedError()

    @abstractmethod
    def step_finished(self, step: Step, result: T) -> T:
        """This method has the opportunity to change the result."""
        raise NotImplementedError()

    @abstractmethod
    def step_failed(self, step: Step, e: Exception) -> None:
        raise NotImplementedError()

    @abstractmethod
    def register_run(self, targets: Iterable[Step], name: Optional[str] = None) -> str:
        raise NotImplementedError()

    @abstractmethod
    def registered_runs(self) -> Dict[str, Dict[str, StepInfo]]:
        raise NotImplementedError()

    def registered_run(self, name: str) -> Dict[str, StepInfo]:
        return self.registered_runs()[name]


@Workspace.register("memory")
class MemoryWorkspace(Workspace):
    def __init__(self):
        self.steps_to_info: Dict[Step, StepInfo] = {}
        self.runs: Dict[str, Set[Step]] = {}

    @property
    def step_cache(self) -> StepCache:
        return step_cache.default_step_cache

    def step_info(self, step: Step) -> StepInfo:
        try:
            return self.steps_to_info[step]
        except KeyError:
            return StepInfo(
                step.unique_id,
                step.name if step.name != step.unique_id else None,
                step.__class__.__name__,
                step.VERSION,
                {dep.unique_id for dep in step.dependencies},
            )

    def steps(
        self, include_completed: bool = True
    ) -> Iterable[StepInfo]:  # TODO: better selection of which steps to return
        if include_completed:
            return self.steps_to_info.values()
        else:
            return (info for info in self.steps_to_info.values() if info.end_time is None)

    def step_started(self, step: Step) -> None:
        self.steps_to_info[step] = StepInfo(
            step.unique_id,
            step.name if step.name != step.unique_id else None,
            step.__class__.__name__,
            step.VERSION,
            {dep.unique_id for dep in step.dependencies},
            datetime.now(),
        )

    def step_finished(self, step: Step, result: T) -> T:
        existing_step_info = self.steps_to_info[step]
        if existing_step_info.status != "running":
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

    def step_failed(self, step: Step, e: Exception) -> None:
        assert e is not None
        existing_step_info = self.steps_to_info[step]
        if existing_step_info.status != "running":
            raise RuntimeError(f"Step {step.name} is failing, but it never started.")
        existing_step_info.end_time = datetime.now()
        existing_step_info.error = e

    def register_run(self, targets: Iterable[Step], name: Optional[str] = None) -> str:
        if name is None:
            name = petname.generate()
        self.runs[name] = set(targets)
        for step in self.runs[name]:
            self.steps_to_info[step] = StepInfo(
                step.unique_id,
                step.name if step.name != step.unique_id else None,
                step.__class__.__name__,
                step.VERSION,
                {dep.unique_id for dep in step.dependencies},
            )
        return name

    def registered_runs(self) -> Dict[str, Dict[str, StepInfo]]:
        return {
            run_name: {step.unique_id: self.steps_to_info[step] for step in steps}
            for run_name, steps in self.runs.items()
        }


default_workspace = MemoryWorkspace()
