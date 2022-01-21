import getpass
import json
import logging
import os
import platform
import socket
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, TypeVar, Union

import petname
from sqlitedict import SqliteDict

from tango.common import FromParams, PathOrStr
from tango.common.file_lock import FileLock
from tango.common.util import exception_to_string
from tango.step import Step
from tango.step_cache import LocalStepCache, StepCache
from tango.version import VERSION
from tango.workspace import Run, StepInfo, StepState, Workspace

logger = logging.getLogger(__name__)

T = TypeVar("T")


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
        except subprocess.CalledProcessError:
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
class ExecutorMetadata(FromParams):
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
                with (run_dir / "conda-environment.yaml").open("w") as f:
                    subprocess.call(["conda", "env", "export"], stdout=f)
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
        self.to_params().to_file(run_dir / "executor-metadata.json")


@Workspace.register("local")
class LocalWorkspace(Workspace):
    """
    This is a :class:`.Workspace` that keeps all its data in a local directory. This works great for single-machine
    jobs, or for multiple machines in a cluster if they can all access the same NFS drive.

    :param dir: The directory to store all the data in

    The directory will have two subdirectories, ``cache/`` for the step cache, and ``runs/`` for the runs. For the
    format of the ``cache/`` directory, refer to :class:`.LocalStepCache`. The ``runs/`` directory will contain one
    subdirectory for each registered run. Each one of those contains a symlink from the name of the step to the
    results directory in the step cache. Note that :class:`.LocalWorkspace` creates these symlinks even for steps
    that have not finished yet. You can tell the difference because either the symlink points to a directory that
    doesn't exist, or it points to a directory in the step cache that doesn't contain results.
    """

    def __init__(self, dir: PathOrStr):
        super().__init__()
        self.dir = Path(dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.cache = LocalStepCache(self.dir / "cache")
        self.locks: Dict[Step, FileLock] = {}
        self.runs_dir = self.dir / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.step_info_file = self.dir / "stepinfo.sqlite"

        # Check the version of the local workspace
        try:
            with open(self.dir / "settings.json", "r") as settings_file:
                settings = json.load(settings_file)
        except FileNotFoundError:
            settings = {"version": 1}

        # Upgrade to version 2
        if settings["version"] == 1:
            with SqliteDict(self.step_info_file) as d:
                for stepinfo_file in self.cache.dir.glob("*/stepinfo.dill"):
                    with stepinfo_file.open("rb") as f:
                        stepinfo = StepInfo.deserialize(f.read())

                    # The `StepInfo` class changed from one version to the next. The deserialized version
                    # ends up being a `StepInfo` instance that is missing the `cacheable` member. This
                    # hack adds it in.
                    kwargs = stepinfo.__dict__
                    kwargs[
                        "cacheable"
                    ] = True  # Only cacheable steps were saved in v1. That's what v2 fixes.
                    d[stepinfo.unique_id] = StepInfo(**kwargs)
                d.commit()
            for stepinfo_file in self.cache.dir.glob("*/stepinfo.dill"):
                stepinfo_file.unlink()

            settings["version"] = 2
            with open(self.dir / "settings.json", "w") as settings_file:
                json.dump(settings, settings_file)

    def step_dir(self, step_or_unique_id: Union[Step, str]) -> Path:
        return self.cache.step_dir(step_or_unique_id)

    @property
    def step_cache(self) -> StepCache:
        return self.cache

    def work_dir(self, step: Step) -> Path:
        result = self.step_dir(step) / "work"
        result.mkdir(parents=True, exist_ok=True)
        return result

    @staticmethod
    def dir_is_empty(dir: Path):
        return not any(True for _ in dir.iterdir())

    @staticmethod
    def _fix_step_info(step_info: StepInfo) -> None:
        """
        Tragically we need to run a fix-up step over StepInfo objects that are freshly read from
        the database. This is for backwards compatibility.

        This function operates on the `step_info` object in place.
        """
        if isinstance(step_info.error, BaseException):
            step_info.error = exception_to_string(step_info.error)

    def step_info(self, step_or_unique_id: Union[Step, str]) -> StepInfo:
        with SqliteDict(self.step_info_file) as d:

            def find_or_add_step_info(step_or_unique_id: Union[Step, str]) -> StepInfo:
                if isinstance(step_or_unique_id, Step):
                    unique_id = step_or_unique_id.unique_id
                else:
                    unique_id = step_or_unique_id

                try:
                    step_info = d[unique_id]
                except KeyError:
                    if not isinstance(step_or_unique_id, Step):
                        raise
                    step = step_or_unique_id

                    for dep in step.dependencies:
                        find_or_add_step_info(dep)

                    step_info = StepInfo(
                        step.unique_id,
                        step.name if step.name != step.unique_id else None,
                        step.__class__.__name__,
                        step.VERSION,
                        {dep.unique_id for dep in step.dependencies},
                        step.cache_results,
                    )
                    d[unique_id] = step_info
                    del step

                # Perform some sanity checks. Sqlite and the file system can get out of sync
                # when a process dies suddenly.
                step_dir = self.step_dir(unique_id)
                new_state = step_info.state
                if not step_dir.exists() or self.dir_is_empty(step_dir):
                    new_state = StepState.INCOMPLETE
                elif step_info.state == StepState.RUNNING and not self._step_lock_file_is_locked(
                    unique_id
                ):
                    new_state = StepState.INCOMPLETE
                if new_state != step_info.state:
                    step_info.start_time = None
                    step_info.end_time = None
                    d[unique_id] = step_info
                return step_info

            result = find_or_add_step_info(step_or_unique_id)
            d.commit()
            self._fix_step_info(result)
            return result

    def _step_lock_file(self, step_or_unique_id: Union[Step, str]) -> Path:
        step_dir = self.step_dir(step_or_unique_id)
        step_dir.mkdir(parents=True, exist_ok=True)
        return step_dir / "lock"

    def _step_lock_file_is_locked(self, step_or_unique_id: Union[Step, str]) -> bool:
        # FileLock.is_locked does not work, so we do this.
        lock = FileLock(self._step_lock_file(step_or_unique_id))
        try:
            lock.acquire(0)
            lock.release()
            return False
        except TimeoutError:
            return True

    def step_starting(self, step: Step) -> None:
        # We don't do anything with uncacheable steps.
        if not step.cache_results:
            return

        # Gather the existing step info first. Step info automatically fixes itself if steps are
        # marked as "running" but are not locked. This happens, for example, when a process
        # gets killed. To make sure this works, we have to get the step info before we start
        # messing with locks.
        step_info = self.step_info(step)
        if step_info.state not in {StepState.INCOMPLETE, StepState.FAILED}:
            raise RuntimeError(
                f"Step '{step.name}' is trying to start, but it is already {step_info.state}. "
                "If you are certain the step is not running somewhere else, delete the lock "
                f"file at {self._step_lock_file(step)}."
            )

        lock = FileLock(self._step_lock_file(step), read_only_ok=True)
        lock.acquire_with_updates(desc=f"acquiring lock for '{step.name}'")
        self.locks[step] = lock

        try:
            step_info.start_time = datetime.now()
            step_info.end_time = None
            step_info.error = None
            step_info.result_location = None
            with SqliteDict(self.step_info_file) as d:
                d[step.unique_id] = step_info
                d.commit()
        except:  # noqa: E722
            lock.release()
            del self.locks[step]
            raise

    def step_finished(self, step: Step, result: T) -> T:
        # We don't do anything with uncacheable steps.
        if not step.cache_results:
            return result

        lock = self.locks[step]

        step_info = self.step_info(step)
        if step_info.state != StepState.RUNNING:
            raise RuntimeError(f"Step '{step.name}' is ending, but it never started.")

        if step.cache_results:
            self.step_cache[step] = result
            if hasattr(result, "__next__"):
                assert isinstance(result, Iterator)
                # Caching the iterator will consume it, so we write it to the cache and then read from the cache
                # for the return value.
                result = self.step_cache[step]

            # Save some metadata.
            def replace_steps_with_unique_id(o: Any):
                if isinstance(o, Step):
                    return {"type": "ref", "ref": o.unique_id}
                if isinstance(o, (list, tuple, set)):
                    return o.__class__(replace_steps_with_unique_id(i) for i in o)
                elif isinstance(o, dict):
                    return {key: replace_steps_with_unique_id(value) for key, value in o.items()}
                else:
                    return o

            try:
                config = step.config
            except ValueError:
                config = None
            metadata = ExecutorMetadata(
                step=step.unique_id, config=replace_steps_with_unique_id(config)
            )
            # Finalize metadata and save to run directory.
            metadata.save(self.step_dir(step))

        # Mark the step as finished
        step_info.end_time = datetime.now()
        step_info.result_location = str(self.step_dir(step).absolute())
        with SqliteDict(self.step_info_file) as d:
            d[step.unique_id] = step_info
            d.commit()

        lock.release()
        del self.locks[step]

        return result

    def step_failed(self, step: Step, e: BaseException) -> None:
        # We don't do anything with uncacheable steps.
        if not step.cache_results:
            return

        lock = self.locks[step]

        try:
            step_info = self.step_info(step)
            if step_info.state != StepState.RUNNING:
                raise RuntimeError(f"Step '{step.name}' is failing, but it never started.")
            step_info.end_time = datetime.now()
            step_info.error = exception_to_string(e)
            with SqliteDict(self.step_info_file) as d:
                d[step.unique_id] = step_info
                d.commit()
        finally:
            lock.release()
            del self.locks[step]

    def register_run(self, targets: Iterable[Step], name: Optional[str] = None) -> Run:
        # sanity check targets
        targets = list(targets)
        for target in targets:
            if not target.cache_results:
                raise RuntimeError(
                    f"Step {target.name} is marked as a target for a run, but is not cacheable. "
                    "Only cacheable steps can be targets."
                )

        if name is None:
            while name is None or (self.runs_dir / name).exists():
                name = petname.generate()
        run_dir = self.runs_dir / name

        # clean any existing run directory
        if run_dir.exists():
            for filename in run_dir.iterdir():
                filename.unlink()
        else:
            run_dir.mkdir(parents=True, exist_ok=True)

        # write step info for all steps
        all_steps = set(targets)
        for step in targets:
            all_steps |= step.recursive_dependencies
        with SqliteDict(self.step_info_file) as d:
            for step in all_steps:
                try:
                    step_info = d[step.unique_id]
                    step_info.name = step.name if step.name != step.unique_id else None
                    d[step.unique_id] = step_info
                except KeyError:
                    d[step.unique_id] = StepInfo(
                        step.unique_id,
                        step.name if step.name != step.unique_id else None,
                        step.__class__.__name__,
                        step.VERSION,
                        {dep.unique_id for dep in step.dependencies},
                        step.cache_results,
                    )
                d.commit()

        # write targets
        for target in targets:
            (run_dir / target.name).symlink_to(os.path.relpath(self.step_dir(target), run_dir))

        return self.registered_run(name)

    def registered_runs(self) -> Dict[str, Run]:
        return {
            str(run_dir.name): self.registered_run(run_dir.name)
            for run_dir in self.runs_dir.iterdir()
            if run_dir.is_dir()
        }

    def registered_run(self, name: str) -> Run:
        run_dir = self.runs_dir / name
        if not run_dir.is_dir():
            raise KeyError(name)
        with SqliteDict(self.step_info_file, flag="r") as d:
            steps_for_run = {}
            for step_symlink in run_dir.iterdir():
                if not step_symlink.is_symlink():
                    continue
                step_name = str(step_symlink.name)
                unique_id = str(step_symlink.resolve().name)
                step_info = d[unique_id]
                assert isinstance(step_info, StepInfo)
                self._fix_step_info(step_info)
                steps_for_run[step_name] = step_info
            return Run(name, steps_for_run, datetime.fromtimestamp(run_dir.stat().st_ctime))

    def run_dir(self, name: str) -> Path:
        """
        Returns the directory where a given run is stored.

        :param name: The name of the run
        :return: The directory where the results of the run are stored

        If the run does not exist, this returns the directory where it will be stored if you call
        :meth:`register_run()` with that name.
        """
        return self.runs_dir / name
