import getpass
import logging
import os
import platform
import socket
import sys
import time
from abc import abstractclassmethod, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from enum import Enum
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
    Set,
    TypeVar,
    Union,
    cast,
)
from urllib.parse import ParseResult, urlparse

import dill

from tango.common import FromParams, Registrable
from tango.step import Step
from tango.step_cache import StepCache
from tango.version import VERSION

logger = logging.getLogger(__name__)

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

    step_name: Optional[str]
    """
    The name of the step, if it has one. Anonymous steps are identified only by their unique ID.

    The same step can have different names in different runs. The last run wins, so don't rely
    on this property in your code. It is just here to aid readability.
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

    cacheable: bool
    """
    Whether or not the step is cacheable.
    """

    start_time: Optional[datetime] = None
    """
    The time this step started running
    """

    end_time: Optional[datetime] = None
    """
    The time this step stopped running. This will be set whether the step succeeded or failed.
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
    A mapping from step names to :class:`StepInfo`, for all the target steps in the run.

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

    @classmethod
    def from_url(cls, url: str) -> "Workspace":
        """
        Initialize a :class:`Workspace` from a workspace URL or path, e.g. ``local:///tmp/workspace``
        would give you a :class:`~tango.workspaces.LocalWorkspace` in the directory ``/tmp/workspace``.

        .. tip::
            Registered as a workspace constructor under the name "from_url".

        """
        parsed = urlparse(url)
        workspace_type = parsed.scheme or "local"
        workspace_cls = cast(Workspace, cls.by_name(workspace_type))
        return workspace_cls.from_parsed_url(parsed)

    @abstractclassmethod
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

        Note that the :class:`Run` object only contains the targets of a run. Usually, that means it
        contains all named steps. Un-named dependencies (or dependencies that are not targets)
        are not contained in the result.

        This method throws ``KeyError`` if there is no run with the given name.
        """
        raise NotImplementedError()

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


@dataclass
class PlatformMetadata(FromParams):
    python: str = field(default_factory=platform.python_version)
    """
    The Python version.
    """

    operating_system: str = field(default_factory=platform.platform)
    """
    Full operating system name.
    """

    executable: Path = field(default_factory=lambda: Path(sys.executable))
    """
    Path to the Python executable.
    """

    cpu_count: Optional[int] = field(default_factory=os.cpu_count)
    """
    Numbers of CPUs on the machine.
    """

    user: str = field(default_factory=getpass.getuser)
    """
    The user that ran this step.
    """

    host: str = field(default_factory=socket.gethostname)
    """
    Name of the host machine.
    """

    root: Path = field(default_factory=lambda: Path(os.getcwd()))
    """
    The root directory from where the Python executable was ran.
    """


@dataclass
class GitMetadata(FromParams):
    commit: Optional[str] = None
    """
    The commit SHA of the current repo.
    """

    remote: Optional[str] = None
    """
    The URL of the primary remote.
    """

    @classmethod
    def check_for_repo(cls) -> Optional["GitMetadata"]:
        import subprocess

        try:
            commit = (
                subprocess.check_output("git rev-parse HEAD".split(" "), stderr=subprocess.DEVNULL)
                .decode("ascii")
                .strip()
            )
            remote: Optional[str] = None
            for line in (
                subprocess.check_output("git remote -v".split(" "))
                .decode("ascii")
                .strip()
                .split("\n")
            ):
                if "(fetch)" in line:
                    _, line = line.split("\t")
                    remote = line.split(" ")[0]
                    break
            return cls(commit=commit, remote=remote)
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None


@dataclass
class TangoMetadata(FromParams):
    version: str = VERSION
    """
    The tango release version.
    """

    command: str = field(default_factory=lambda: " ".join(sys.argv))
    """
    The exact command used.
    """


@dataclass
class StepExecutionMetadata(FromParams):
    """
    Represents data collected during the execution of a step. This class can be used by :class:`Workspace`
    implementations to store this data for each step.
    """

    step: str
    """
    The unique ID of the step.
    """

    config: Optional[Dict[str, Any]] = None
    """
    The raw config of the step.
    """

    platform: PlatformMetadata = field(default_factory=PlatformMetadata)
    """
    The :class:`PlatformMetadata`.
    """

    git: Optional[GitMetadata] = field(default_factory=GitMetadata.check_for_repo)
    """
    The :class:`GitMetadata`.
    """

    tango: Optional[TangoMetadata] = field(default_factory=TangoMetadata)
    """
    The :class:`TangoMetadata`.
    """

    started_at: float = field(default_factory=time.time)
    """
    The unix timestamp from when the run was started.
    """

    finished_at: Optional[float] = None
    """
    The unix timestamp from when the run finished.
    """

    duration: Optional[float] = None
    """
    The number of seconds the step ran for.
    """

    def _save_pip(self, run_dir: Path):
        """
        Saves the current working set of pip packages to ``run_dir``.
        """
        # Adapted from the Weights & Biases client library:
        # github.com/wandb/client/blob/a04722575eee72eece7eef0419d0cea20940f9fe/wandb/sdk/internal/meta.py#L56-L72
        try:
            import pkg_resources

            installed_packages = [d for d in iter(pkg_resources.working_set)]
            installed_packages_list = sorted(
                ["%s==%s" % (i.key, i.version) for i in installed_packages]
            )
            with (run_dir / "requirements.txt").open("w") as f:
                f.write("\n".join(installed_packages_list))
        except Exception as exc:
            logger.exception("Error saving pip packages: %s", exc)

    def _save_conda(self, run_dir: Path):
        """
        Saves the current conda environment to ``run_dir``.
        """
        # Adapted from the Weights & Biases client library:
        # github.com/wandb/client/blob/a04722575eee72eece7eef0419d0cea20940f9fe/wandb/sdk/internal/meta.py#L74-L87
        current_shell_is_conda = os.path.exists(os.path.join(sys.prefix, "conda-meta"))
        if current_shell_is_conda:
            import subprocess

            try:
                result = subprocess.run(["conda", "env", "export"], capture_output=True)
                if (
                    result.returncode != 0
                    and result.stderr is not None
                    and "Unable to determine environment" in result.stderr.decode()
                ):
                    result = subprocess.run(
                        ["conda", "env", "export", "-n", "base"], capture_output=True
                    )

                if result.returncode != 0:
                    if result.stderr is not None:
                        logger.exception("Error saving conda packages: %s", result.stderr.decode())
                    else:
                        result.check_returncode()
                elif result.stdout is not None:
                    with (run_dir / "conda-environment.yaml").open("w") as f:
                        f.write(result.stdout.decode())
            except Exception as exc:
                logger.exception("Error saving conda packages: %s", exc)

    def save(self, run_dir: Path):
        """
        Should be called after the run has finished to save to file.
        """
        self.finished_at = time.time()
        self.duration = round(self.finished_at - self.started_at, 4)

        # Save pip dependencies and conda environment files.
        self._save_pip(run_dir)
        self._save_conda(run_dir)

        # Serialize self.
        self.to_params().to_file(run_dir / "execution-metadata.json")
