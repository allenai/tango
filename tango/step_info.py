import getpass
import logging
import os
import platform
import socket
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pytz

from .common.from_params import FromParams
from .common.logging import log_exception
from .common.util import jsonify, local_timezone, replace_steps_with_unique_id
from .step import Step
from .version import VERSION

logger = logging.getLogger(__name__)


def get_pip_packages() -> Optional[List[Tuple[str, str]]]:
    """
    Get the current working set of pip packages. Equivalent to running ``pip freeze``.
    """
    # Adapted from the Weights & Biases client library:
    # github.com/wandb/client/blob/a04722575eee72eece7eef0419d0cea20940f9fe/wandb/sdk/internal/meta.py#L56-L72
    try:
        import pkg_resources

        return sorted([(d.key, d.version) for d in iter(pkg_resources.working_set)])
    except Exception as exc:
        logger.error("Error saving pip packages")
        log_exception(exc)
    return None


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
        from git import InvalidGitRepositoryError, Repo

        try:
            repo = Repo(".")
        except InvalidGitRepositoryError:
            return None

        return cls(commit=str(repo.commit()), remote=repo.remote().url)


@dataclass
class TangoMetadata(FromParams):
    version: str = VERSION
    """
    The tango release version.
    """


@dataclass
class PlatformMetadata(FromParams):
    operating_system: str = field(default_factory=platform.platform)
    """
    Full operating system name.
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


@dataclass
class EnvironmentMetadata(FromParams):
    python: str = field(default_factory=platform.python_version)
    """
    The Python version.
    """

    executable: Path = field(default_factory=lambda: Path(sys.executable))
    """
    Path to the Python executable.
    """

    command: str = field(default_factory=lambda: " ".join(sys.argv))
    """
    The exact command used.
    """

    root: Path = field(default_factory=lambda: Path(os.getcwd()))
    """
    The root directory from where the Python executable was ran.
    """

    packages: Optional[List[Tuple[str, str]]] = field(default_factory=get_pip_packages)
    """
    The current set of Python packages in the Python environment. Each entry is a tuple of strings.
    The first element is the name of the package, the second element is the version.
    """

    git: Optional[GitMetadata] = field(default_factory=GitMetadata.check_for_repo)
    """
    The :class:`GitMetadata`.
    """

    tango: Optional[TangoMetadata] = field(default_factory=TangoMetadata)
    """
    The :class:`TangoMetadata`.
    """


@dataclass
class StepInfo(FromParams):
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

    config: Optional[Dict[str, Any]] = None
    """
    The raw config of the step.
    """

    metadata: Optional[Dict[str, Any]] = None
    """
    Metadata from the step. This comes from the ``step_metadata``
    argument to the :class:`~tango.step.Step` class.
    """

    platform: PlatformMetadata = field(default_factory=PlatformMetadata)
    """
    The :class:`PlatformMetadata`.
    """

    environment: EnvironmentMetadata = field(default_factory=EnvironmentMetadata)
    """
    The :class:`EnvironmentMetadata`.
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
        return jsonify(self)

    @classmethod
    def from_json_dict(cls, json_dict: Dict[str, Any]) -> "StepInfo":
        """
        The inverse of :meth:`to_json_dict()`.

        :param json_dict: A dictionary representation, such as the one produced by :meth:`to_json_dict()`.
        """
        return cls.from_params(
            {
                k: (
                    datetime.strptime(v, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=pytz.utc)
                    if k in {"start_time", "end_time"} and v is not None
                    else v
                )
                for k, v in json_dict.items()
            }
        )

    @classmethod
    def new_from_step(cls, step: Step, **kwargs) -> "StepInfo":
        try:
            config = step.config
        except ValueError:
            config = None
        return cls(
            unique_id=step.unique_id,
            step_name=step.name,
            step_class_name=step.__class__.__name__,
            version=step.VERSION,
            dependencies={dep.unique_id for dep in step.dependencies},
            cacheable=step.cache_results,
            config=replace_steps_with_unique_id(config),
            metadata=step.metadata,
            **kwargs,
        )

    def refresh(self):
        """
        Refresh environment and platform metadata.
        """
        self.platform = PlatformMetadata()
        self.environment = EnvironmentMetadata()
