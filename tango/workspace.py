import logging
from abc import abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import (
    Any,
    ContextManager,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    TypeVar,
    Union,
    cast,
)
from urllib.parse import ParseResult, urlparse

from .common import Registrable
from .step import Step
from .step_cache import StepCache
from .step_info import StepInfo

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class Run:
    """
    Stores information about a single Tango run.
    """

    name: str
    """
    The name of the run
    """

    steps: Dict[str, StepInfo]
    """
    A mapping from step names to :class:`~tango.step_info.StepInfo`, for all the target steps in the run.

    This only contains the targets of a run. Usually, that means it contains all named steps.
    Un-named dependencies (or dependencies that are not targets) are not contained in ``steps``.
    """

    start_date: datetime
    """
    The time at which the run was registered in the workspace.
    """


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

    def __init__(self):
        self._delayed_cleanup_temp_dirs: List[TemporaryDirectory] = []

    def __getstate__(self):
        """
        We override `__getstate__()` to customize how instances of this class are pickled
        since we don't want to persist certain attributes.
        """
        out = {k: v for k, v in self.__dict__.items() if k not in {"_delayed_cleanup_temp_dirs"}}
        out["_delayed_cleanup_temp_dirs"] = []
        return out

    @property
    @abstractmethod
    def url(self) -> str:
        """
        Get a URL for the workspace that can be used to instantiate the same workspace
        using :meth:`.from_url()`.
        """
        raise NotImplementedError

    @classmethod
    def from_url(cls, url: str) -> "Workspace":
        """
        Initialize a :class:`Workspace` from a workspace URL or path, e.g. ``local:///tmp/workspace``
        would give you a :class:`~tango.workspaces.LocalWorkspace` in the directory ``/tmp/workspace``.

        For :class:`~tango.workspaces.LocalWorkspace`, you can also just pass in a plain
        path, e.g. ``/tmp/workspace``.

        .. tip::
            Registered as a workspace constructor under the name "from_url".

        """
        parsed = urlparse(url)
        workspace_type = parsed.scheme or "local"
        workspace_cls = cast(Workspace, cls.by_name(workspace_type))
        return workspace_cls.from_parsed_url(parsed)

    @classmethod
    @abstractmethod
    def from_parsed_url(cls, parsed_url: ParseResult) -> "Workspace":
        """
        Subclasses should override this so that can be initialized from a URL.

        :param parsed_url: The parsed URL object.
        """
        raise NotImplementedError

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

        # TemporaryDirectory cleans up the directory automatically when the TemporaryDirectory object
        # gets garbage collected, so we hold on to it in the Workspace.
        dir = TemporaryDirectory(prefix=f"{step.unique_id}-", suffix=".step_dir")
        self._delayed_cleanup_temp_dirs.append(dir)
        return Path(dir.name)

    @abstractmethod
    def step_info(self, step_or_unique_id: Union[Step, str]) -> StepInfo:
        """
        Returns a :class:`~tango.step_info.StepInfo` for a given step.

        :raises KeyError: If the corresponding step info cannot be found or created.
            This should never happen if you pass a :class:`~tango.step.Step` object to this method
            since a :class:`~tango.step_info.StepInfo` can always be created from a
            :class:`~tango.step.Step`.
        """
        raise NotImplementedError()

    @abstractmethod
    def step_starting(self, step: Step) -> None:
        """
        The :class:`.Step` class calls this when a step is about to start running.

        :param step: The step that is about to start.

        :raises StepStateError: If the step is in an unexpected state (e.g. RUNNING).
        """
        raise NotImplementedError()

    @abstractmethod
    def step_finished(self, step: Step, result: T) -> T:
        """
        The :class:`.Step` class calls this when a step finished running.

        :param step: The step that finished.

        :raises StepStateError: If the step is in an unexpected state (e.g. RUNNING).

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

        :raises StepStateError: If the step is in an unexpected state (e.g. RUNNING).
        """
        raise NotImplementedError()

    @abstractmethod
    def register_run(self, targets: Iterable[Step], name: Optional[str] = None) -> Run:
        """
        Register a run in the workspace. A run is a set of target steps that a user wants to execute.

        :param targets: The steps that the user wants to execute. This could come from a :class:`.StepGraph`.
        :param name: A name for the run. Runs must have unique names. If not given, this method invents a name and
                     returns it.
        :return: The run object
        """
        raise NotImplementedError()

    @abstractmethod
    def registered_runs(self) -> Dict[str, Run]:
        """
        Returns all runs in the workspace

        :return: A dictionary mapping run names to :class:`Run` objects
        """
        raise NotImplementedError()

    @abstractmethod
    def registered_run(self, name: str) -> Run:
        """
        Returns the run with the given name

        :return: A :class:`Run` object representing the named run

        :raises KeyError: If there is no run with the given name.
        """
        raise NotImplementedError()

    def step_result_for_run(self, run_name: str, step_name: str) -> Any:
        """
        Get the result of a step from a run.

        :raises KeyError: If there is no run or step with the given name.
        """
        run = self.registered_run(run_name)
        step_info = run.steps[step_name]
        try:
            return self.step_cache[step_info]
        except KeyError:
            raise KeyError(f"Step result for '{step_name}' not found in workspace")

    def step_result(self, step_name: str) -> Any:
        """
        Get the result of a step from the latest run with a step by that name.

        :raises KeyError: If there is no run with the given step.
        """
        runs = sorted(self.registered_runs().values(), key=lambda run: run.start_date, reverse=True)
        for run in runs:
            if step_name in run.steps:
                return self.step_cache[run.steps[step_name]]
        raise KeyError(f"No step named '{step_name}' found in previous runs")

    def capture_logs_for_run(self, name: str) -> ContextManager[None]:
        """
        Should return a context manager that can be used to capture the logs for a run.

        By default, this doesn't do anything.

        Examples
        --------

        The :class:`.LocalWorkspace` implementation uses this method to capture logs
        to a file in the workspace's directory using the :func:`~tango.common.logging.file_handler()`
        context manager, similar to this:

        .. testcode::

            from tango.common.logging import file_handler
            from tango.workspace import Workspace

            class MyLocalWorkspace(Workspace):
                def capture_logs_for_run(self, name: str):
                    return file_handler("/path/to/workspace/" + name + ".log")

        """

        @contextmanager
        def do_nothing() -> Generator[None, None, None]:
            yield None

        return do_nothing()


Workspace.register("from_url", constructor="from_url")
