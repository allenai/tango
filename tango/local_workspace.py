import getpass
import logging
import os
import platform
import socket
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, TypeVar, Union

import petname

from tango.common import FromParams, PathOrStr
from tango.common.file_lock import FileLock
from tango.step import Step
from tango.step_cache import LocalStepCache, StepCache
from tango.version import VERSION
from tango.workspace import StepInfo, StepState, Workspace

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
        self.dir = Path(dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.cache = LocalStepCache(self.dir / "cache")
        self.locks: Dict[Step, FileLock] = {}
        self.runs_dir = self.dir / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def step_dir(self, step_or_unique_id: Union[Step, str]) -> Path:
        return self.cache.step_dir(step_or_unique_id)

    @property
    def step_cache(self) -> StepCache:
        return self.cache

    def work_dir(self, step: Step) -> Path:
        result = self.step_dir(step) / "work"
        result.mkdir(parents=True, exist_ok=True)
        return result

    def _step_info_file(self, step_or_unique_id: Union[Step, str]) -> Path:
        return self.step_dir(step_or_unique_id) / "stepinfo.dill"

    def step_info(self, step_or_unique_id: Union[Step, str]) -> StepInfo:
        return self._get_step_info(step_or_unique_id)

    def _get_step_info(self, step_or_unique_id: Union[Step, str]) -> StepInfo:
        try:
            with self._step_info_file(step_or_unique_id).open("rb") as f:
                return StepInfo.deserialize(f.read())
        except FileNotFoundError:
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

    def _put_step_info(self, step: Step, step_info: StepInfo) -> None:
        path = self._step_info_file(step)
        try:
            path.parent.mkdir()
        except FileExistsError:
            pass
        with path.open("wb") as f:
            dump = step_info.serialize()
            f.write(dump)

    def _step_lock_file(self, step: Step) -> Path:
        step_dir = self.step_dir(step)
        step_dir.mkdir(parents=True, exist_ok=True)
        return step_dir / "lock"

    def step_starting(self, step: Step) -> None:
        lock = FileLock(self._step_lock_file(step), read_only_ok=True)
        lock.acquire_with_updates(desc=f"acquiring lock for {step.name}")
        self.locks[step] = lock

        try:
            step_info = self._get_step_info(step)
            if step_info.state not in {StepState.INCOMPLETE, StepState.FAILED}:
                raise RuntimeError(
                    f"Step {step.name} is trying to start, but it is already {step_info.state}."
                )

            step_info.start_time = datetime.now()
            step_info.end_time = None
            step_info.error = None
            step_info.result_location = None
            self._put_step_info(step, step_info)
        except:  # noqa: E722
            lock.release()
            del self.locks[step]
            raise

    def step_finished(self, step: Step, result: T) -> T:
        lock = self.locks[step]

        step_info = self._get_step_info(step)
        if step_info.state != StepState.RUNNING:
            raise RuntimeError(f"Step {step.name} is ending, but it never started.")

        if step.cache_results:
            self.step_cache[step] = result
            if hasattr(result, "__next__"):
                assert isinstance(result, Iterator)
                # Caching the iterator will consume it, so we write it to the cache and then read from the cache
                # for the return value.
                result = self.step_cache[step]

        step_info.end_time = datetime.now()
        step_info.result_location = str(self.step_dir(step).absolute())
        self._put_step_info(step, step_info)

        # Initialize metadata.
        def replace_steps_with_unique_id(o: Any):
            if isinstance(o, Step):
                return {"type": "ref", "ref": o.unique_id}
            if isinstance(o, (list, tuple, set)):
                return o.__class__(replace_steps_with_unique_id(i) for i in o)
            elif isinstance(o, dict):
                return {key: replace_steps_with_unique_id(value) for key, value in o.items()}
            else:
                return o

        metadata = ExecutorMetadata(
            step=step.unique_id, config=replace_steps_with_unique_id(step.config)
        )
        # Finalize metadata and save to run directory.
        metadata.save(self.step_dir(step))

        lock.release()
        del self.locks[step]

        return result

    def step_failed(self, step: Step, e: BaseException) -> None:
        step_info = self._get_step_info(step)
        if step_info.state != StepState.RUNNING:
            raise RuntimeError(f"Step {step.name} is failing, but it never started.")
        step_info.end_time = datetime.now()
        step_info.error = e
        self._put_step_info(step, step_info)

    def register_run(self, targets: Iterable[Step], name: Optional[str] = None) -> str:
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
            all_steps.union(step.recursive_dependencies)
        for step in all_steps:
            self._put_step_info(step, self._get_step_info(step))

        # write targets
        for target in targets:
            (run_dir / target.name).symlink_to(os.path.relpath(self.step_dir(target), run_dir))

        return name

    def registered_runs(self) -> List[str]:
        return [str(run_dir.name) for run_dir in self.runs_dir.iterdir() if run_dir.is_dir()]

    def registered_run(self, name: str) -> Dict[str, StepInfo]:
        run_dir = self.runs_dir / name
        steps_for_run = {}
        for step_symlink in run_dir.iterdir():
            if not step_symlink.is_symlink():
                continue
            step_name = str(step_symlink.name)
            unique_id = str(step_symlink.resolve().name)
            step_info = self._get_step_info(unique_id)
            assert isinstance(step_info, StepInfo)
            steps_for_run[step_name] = step_info
        return steps_for_run

    def run_dir(self, name: str) -> Path:
        """
        Returns the directory where a given run is stored.

        :param name: The name of the run
        :return: The directory where the results of the run are stored

        If the run does not exist, this returns the directory where it will be stored if you call
        :meth:`register_run()` with that name.
        """
        return self.runs_dir / name
