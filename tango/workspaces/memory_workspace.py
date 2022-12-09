import copy
from fnmatch import fnmatch
from typing import Dict, Generator, Iterable, Iterator, Optional, TypeVar, Union
from urllib.parse import ParseResult

import petname

from tango.common.exceptions import StepStateError
from tango.common.util import exception_to_string, utc_now_datetime
from tango.step import Step
from tango.step_cache import StepCache
from tango.step_caches import default_step_cache
from tango.step_info import StepInfo, StepState
from tango.workspace import Run, StepInfoSort, Workspace, WorkspaceSort

T = TypeVar("T")


@Workspace.register("memory")
class MemoryWorkspace(Workspace):
    """
    This is a workspace that keeps all its data in memory. This is useful for debugging or for quick jobs, but of
    course you don't get any caching across restarts.

    .. tip::

        Registered as a :class:`~tango.workspace.Workspace` under the name "memory".
    """

    def __init__(self):
        super().__init__()
        self.unique_id_to_info: Dict[str, StepInfo] = {}
        self.runs: Dict[str, Run] = {}

    @property
    def url(self) -> str:
        return "memory://"

    @classmethod
    def from_parsed_url(cls, parsed_url: ParseResult) -> "Workspace":
        return cls()

    @property
    def step_cache(self) -> StepCache:
        return default_step_cache

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
                return StepInfo.new_from_step(step)
            else:
                raise KeyError()

    def step_starting(self, step: Step) -> None:
        # We don't do anything with uncacheable steps.
        if not step.cache_results:
            return

        self.unique_id_to_info[step.unique_id] = StepInfo.new_from_step(
            step, start_time=utc_now_datetime()
        )

    def step_finished(self, step: Step, result: T) -> T:
        # We don't do anything with uncacheable steps.
        if not step.cache_results:
            return result

        existing_step_info = self.unique_id_to_info[step.unique_id]
        if existing_step_info.state != StepState.RUNNING:
            raise StepStateError(step, existing_step_info.state)
        existing_step_info.end_time = utc_now_datetime()

        if step.cache_results:
            self.step_cache[step] = result
            if hasattr(result, "__next__"):
                assert isinstance(result, Iterator)
                # Caching the iterator will consume it, so we write it to the cache and then read from the cache
                # for the return value.
                return self.step_cache[step]
        return result

    def step_failed(self, step: Step, e: BaseException) -> None:
        # We don't do anything with uncacheable steps.
        if not step.cache_results:
            return

        assert e is not None
        existing_step_info = self.unique_id_to_info[step.unique_id]
        if existing_step_info.state != StepState.RUNNING:
            raise StepStateError(step, existing_step_info.state)
        existing_step_info.end_time = utc_now_datetime()
        existing_step_info.error = exception_to_string(e)

    def register_run(self, targets: Iterable[Step], name: Optional[str] = None) -> Run:
        if name is None:
            name = petname.generate()
        steps: Dict[str, StepInfo] = {}
        for step in targets:
            step_info = StepInfo.new_from_step(step)
            self.unique_id_to_info[step.unique_id] = step_info
            steps[step.unique_id] = step_info
        run = Run(name, steps, utc_now_datetime())
        self.runs[name] = run
        return run

    def search_step_info(
        self,
        *,
        sort_by: StepInfoSort = StepInfoSort.START_TIME,
        sort_descending: bool = True,
        match: Optional[str] = None,
        limit: Optional[int] = None,
        cursor: Optional[int] = None,
    ) -> Generator[StepInfo, None, None]:
        steps = [
            step
            for step in self.unique_id_to_info.values()
            if match is None or fnmatch(step.unique_id, match)
        ]

        if sort_by == StepInfoSort.START_TIME:
            now = utc_now_datetime()
            steps = sorted(steps, key=lambda step: step.start_time or now, reverse=sort_descending)
        elif sort_by == StepInfoSort.UNIQUE_ID:
            steps = sorted(steps, key=lambda step: step.unique_id, reverse=sort_descending)
        else:
            raise NotImplementedError

        if cursor is not None:
            steps = steps[cursor:]
        if limit:
            steps = steps[:limit]

        yield from steps

    def search_registered_runs(
        self,
        *,
        sort_by: WorkspaceSort = WorkspaceSort.START_DATE,
        sort_descending: bool = True,
        match: Optional[str] = None,
        limit: Optional[int] = None,
        cursor: Optional[int] = None,
    ) -> Generator[Run, None, None]:
        if match is not None:
            runs = [run for run in self.runs.values() if fnmatch(run.name, match)]
        else:
            runs = list(self.runs.values())

        if sort_by == WorkspaceSort.START_DATE:
            runs = sorted(runs, key=lambda run: run.start_date, reverse=sort_descending)
        elif sort_by == WorkspaceSort.NAME:
            runs = sorted(runs, key=lambda run: run.name, reverse=sort_descending)
        else:
            raise NotImplementedError

        if cursor is not None:
            runs = runs[cursor:]
        if limit:
            runs = runs[:limit]

        yield from runs

    def registered_run(self, name: str) -> Run:
        return copy.deepcopy(self.runs[name])


default_workspace = MemoryWorkspace()
