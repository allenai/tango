import getpass
import logging
import os
import platform
import socket
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar

import click

from tango.common.aliases import PathOrStr
from tango.common.file_lock import FileLock
from tango.common.from_params import FromParams
from tango.common.util import import_extra_module
from tango.step import Step
from tango.step_cache import StepCache
from tango.step_graph import StepGraph
from tango.version import VERSION

logger = logging.getLogger(__name__)


T = TypeVar("T")


class Executor:
    """
    An ``Executor`` is a class that is responsible for running steps and caching their results.
    """

    def __init__(
        self, dir: PathOrStr, step_cache: StepCache, include_package: Optional[List[str]] = None
    ) -> None:
        self.dir = Path(dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.step_cache = step_cache
        self.include_package = include_package

    def execute_step_graph(self, step_graph: StepGraph) -> None:
        """
        Execute a :class:`tango.step_graph.StepGraph`.
        """
        # Import included packages to find registered components.
        if self.include_package is not None:
            for package_name in self.include_package:
                import_extra_module(package_name)

        # Acquire lock on directory to make sure no other Executors are writing
        # to it at the same time.
        directory_lock = FileLock(self.dir / ".lock", read_only_ok=True)
        directory_lock.acquire_with_updates(desc="acquiring directory lock...")

        try:
            # Remove symlinks to old results.
            for filename in self.dir.glob("*"):
                if filename.is_symlink():
                    relative_target = os.readlink(filename)
                    if not relative_target.startswith("step_cache/"):
                        continue
                    logger.debug(
                        f"Removing symlink '{filename.name}' to previous result {relative_target}"
                    )
                    filename.unlink()

            ordered_steps = sorted(step_graph.values(), key=lambda step: step.name)

            # Determine which steps we actually need to run, i.e. steps that fall into one of the
            # following two categories:
            #  1. step should be cached but is not in cache
            #  2. step is a dependency (direct or recursively) to another step that should be cached
            #     but is not in the cache.
            for step in ordered_steps:
                if step.cache_results:
                    self.execute_step_with_dependencies(step)

            # Symlink everything that has been computed.
            for step in ordered_steps:
                name = step.name
                if step in self.step_cache:
                    step_link = self.dir / name
                    if step_link.exists():
                        step_link.unlink()
                    step_link.symlink_to(
                        self.step_dir(step).relative_to(self.dir),
                        target_is_directory=True,
                    )
                    click.echo(
                        click.style("\N{check mark} The output for ", fg="green")
                        + click.style(f'"{name}"', bold=True, fg="green")
                        + click.style(" is in ", fg="green")
                        + click.style(f"{step_link}", bold=True, fg="green")
                    )
        finally:
            # Release lock on directory.
            directory_lock.release()

    def execute_step_with_dependencies(
        self, step: Step[T], needed_by: Optional[Step] = None
    ) -> Optional[T]:
        """
        Runs a step and all of it's dependencies.
        """
        if step not in self.step_cache:
            # We'll keep track of the results for dependencies that can't be
            # added to the step cache in this dictionary (kind of like a temporary
            # in-memory step cache) and then inject them into the step's kwargs
            # before running.
            non_cacheable_dependencies: Dict[Step, Any] = {}
            for dependency in step.dependencies:
                maybe_result = self.execute_step_with_dependencies(dependency, needed_by=step)
                if not dependency.cache_results:
                    non_cacheable_dependencies[dependency] = maybe_result
            step._inject_dependencies(non_cacheable_dependencies)
        return self.execute_step(step, needed_by=needed_by)

    def execute_step(
        self, step: Step[T], quiet: bool = False, needed_by: Optional[Step] = None
    ) -> Optional[T]:
        """
        Execute a step. Assumes all step dependencies have already been resolved.
        If the step is not cacheable, this will return the result of the step.
        """
        # Prepare directory and acquire lock.
        run_dir = self.step_dir(step)
        run_dir.mkdir(parents=True, exist_ok=True)
        run_dir_lock = FileLock(run_dir / ".lock", read_only_ok=True)
        run_dir_lock.acquire_with_updates(desc="acquiring run dir lock...")

        if step in self.step_cache:
            if not quiet:
                click.echo(
                    click.style("\N{check mark} Found output for ", fg="green")
                    + click.style(f'"{step.name}"', bold=True, fg="green")
                    + click.style(
                        " in cache" + ""
                        if needed_by is None
                        else f' (needed by "{needed_by.name}")',
                        fg="green",
                    )
                )
            run_dir_lock.release()
            return None

        if not quiet:
            click.echo(
                click.style("\N{black circle} Starting run for ", fg="blue")
                + click.style(f'"{step.name}"', bold=True, fg="blue")
                + click.style(
                    "..." if needed_by is None else f' (needed by "{needed_by.name}")...', fg="blue"
                )
            )

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

        maybe_result: Optional[T] = None
        try:
            # Run the step.
            if step.cache_results:
                step.ensure_result(self.step_cache)
            else:
                maybe_result = step.result(self.step_cache)

            # Finalize metadata and save to run directory.
            metadata.save(run_dir)
        finally:
            # Release lock on run dir.
            run_dir_lock.release()

        if not quiet:
            click.echo(
                click.style("\N{check mark} Finished run for ", fg="green")
                + click.style(f'"{step.name}"', bold=True, fg="green")
            )

        return maybe_result

    def step_dir(self, step: Step) -> Path:
        """
        Returns a unique directory to use for the run of the given step.

        This is the :class:`~pathlib.Path` returned by :meth:`~tango.step_cache.work_dir()`.
        """
        return self.step_cache.step_dir(step)


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
