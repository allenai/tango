from abc import abstractmethod
from dataclasses import dataclass, field
import getpass
import logging
import os
from pathlib import Path
import platform
import time
from typing import List, Tuple, Iterable, Any, Dict, Optional
import socket
import sys

import click

from tango.common.file_lock import FileLock
from tango.common.from_params import FromParams
from tango.common.params import Params
from tango.common.registrable import Registrable
from tango.common.util import PathOrStr, import_module_and_submodules
from tango.step import Step
from tango.step_cache import StepCache
from tango.step_graph import StepGraph, StepStub


logger = logging.getLogger(__name__)


class Executor(Registrable):
    """
    An ``Executor`` is a :class:`~tango.common.registrable.Registrable` class that is
    responsible for running steps and returning them with their result.

    Subclasses should implement :meth:`execute_step_group()`.
    """

    default_implementation = "simple"
    """
    The default implementation is :class:`SimpleExecutor`.
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
        Execute an entire :class:`tango.step_graph.StepGraph`.
        """
        # Import included packages to find registered components.
        if self.include_package is not None:
            for package_name in self.include_package:
                import_module_and_submodules(package_name)

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

            # Keeps track of all steps that we've ran so far by name, and stores the results
            # for ones that can't be cached.
            executed: Dict[str, Tuple[Step, Any]] = {}
            remaining: List[StepStub] = list(step_graph)
            group: List[Step] = []
            all_steps: List[Step] = []

            def run_group():
                # Gather all steps that need to be ran (no cache hit).
                needed: List[Step] = []
                for step in group:
                    if step not in self.step_cache:
                        needed.append(step)
                    else:
                        executed[step.name] = (step, None)
                        click.echo(
                            click.style("✓ Found output for ", fg="green")
                            + click.style(f'"{step.name}"', bold=True, fg="green")
                            + click.style(" in cache", fg="green")
                        )
                if needed:
                    # Run the needed ones.
                    for step, result in self.execute_step_group(needed):
                        # Add to cache or keep result in `executed`.
                        if step.cache_results and step not in self.step_cache:
                            self.step_cache[step] = result
                            executed[step.name] = (step, None)
                        else:
                            executed[step.name] = (step, result)

            while remaining:
                next_step_ready = True
                for ref in remaining[0].dependencies:
                    if ref not in executed:
                        # Still has dependencies that need to be ran first.
                        next_step_ready = False
                        break

                if not next_step_ready:
                    # Run the current group.
                    assert group
                    run_group()
                    all_steps.extend(group)
                    group = []

                # Materialize the next step.
                next_up = remaining.pop(0)
                config = self._replace_refs_with_results(next_up.config, executed, self.step_cache)
                step = Step.from_params(Params(config), step_name=next_up.name)
                step._executor = self
                group.append(step)

            # Finish up last group.
            if group:
                run_group()
                all_steps.extend(group)

            # Symlink everything that has been computed.
            for step in all_steps:
                name = step.name
                if step in self.step_cache:
                    step_link = self.dir / name
                    if step_link.exists():
                        step_link.unlink()
                    step_link.symlink_to(
                        self.directory_for_run(step).relative_to(self.dir),
                        target_is_directory=True,
                    )
                    click.echo(
                        click.style("✓ The output for ", fg="green")
                        + click.style(f'"{name}"', bold=True, fg="green")
                        + click.style(" is in ", fg="green")
                        + click.style(f"{step_link}", bold=True, fg="green")
                    )
        finally:
            # Release lock on directory.
            directory_lock.release()

    @abstractmethod
    def execute_step_group(self, step_group: List[Step]) -> Iterable[Tuple[Step, Any]]:
        """
        Execute all steps in the group, returning them with their results.

        The executor can assume that all steps in the group are independent (none of them
        depend on the result of any other step in the group), so they can be ran
        in any order or even in parallel.

        When implementing this method it might make sense to use :meth:`execute_step()`.
        """
        raise NotImplementedError

    def execute_step(self, step: Step, quiet: bool = False) -> Any:
        """
        This method is provided for convenience. It is a robust way to run a step
        that will acquire a lock on the step's run directory and ensure metadata
        is saved after the run.

        It can be used internally by subclasses in their :meth:`execute_step_group()` method.
        """
        if not quiet:
            click.echo(
                click.style("● Starting run for ", fg="blue")
                + click.style(f'"{step.name}"', bold=True, fg="blue")
            )

        # Initialize metadata.
        metadata = ExecutorMetadata(step=step.unique_id)

        # Prepare directory and acquire lock.
        run_dir = self.directory_for_run(step)
        run_dir.mkdir(parents=True, exist_ok=True)
        run_dir_lock = FileLock(run_dir / ".lock", read_only_ok=True)
        run_dir_lock.acquire_with_updates(desc="acquiring run dir lock...")

        try:
            # Run the step.
            result = step.run_with_work_dir(run_dir / "work")

            # Finalize metadata and save to run directory.
            metadata.finalize()
            metadata.to_params().to_file(run_dir / "executor-metadata.json")
        finally:
            # Release lock on run dir.
            run_dir_lock.release()

        if not quiet:
            click.echo(
                click.style("✓ Finished run for ", fg="green")
                + click.style(f'"{step.name}"', bold=True, fg="green")
            )

        return result

    def directory_for_run(self, step: Step) -> Path:
        """
        Returns a unique directory to use for the run of the given step.

        This is the :class:`~pathlib.Path` returned by :meth:`~tango.step_cache.directory_for_run()`.
        """
        return self.step_cache.directory_for_run(step)

    @staticmethod
    def _replace_refs_with_results(
        o: Any, executed: Dict[str, Tuple[Step, Any]], step_cache: StepCache
    ) -> Any:
        if isinstance(o, list):
            return [Executor._replace_refs_with_results(x, executed, step_cache) for x in o]
        elif isinstance(o, tuple):
            return tuple(Executor._replace_refs_with_results(list(o), executed, step_cache))
        elif isinstance(o, set):
            return set(Executor._replace_refs_with_results(list(o), executed, step_cache))
        elif isinstance(o, dict):
            if set(o.keys()) == {"type", "ref"}:
                if o["ref"] in executed:
                    step, result = executed[o["ref"]]
                    if result is not None:
                        return result
                    else:
                        return step_cache[step]
                else:
                    raise ValueError(f"result for step '{o['ref']}' could not be found!")
            else:
                return {
                    k: Executor._replace_refs_with_results(v, executed, step_cache)
                    for k, v in o.items()
                }
        elif o is None or isinstance(o, (str, bool, int, float)):
            return o
        else:
            raise ValueError(o)


@Executor.register("simple")
class SimpleExecutor(Executor):
    """
    A simple :class:`Executor` that just runs all steps locally one at a time.

    .. tip::
        Registered as an :class:`Executor` under the name "simple".

    """

    def execute_step_group(self, step_group: List[Step]) -> Iterable[Tuple[Step, Any]]:
        for step in step_group:
            result = self.execute_step(step)
            yield step, result


@dataclass
class PlatformMetadata(FromParams):
    python: str = platform.python_version()
    operating_system: str = platform.platform()
    executable: Path = Path(sys.executable)
    cpu_count: Optional[int] = os.cpu_count()
    user: str = getpass.getuser()
    host: str = socket.gethostname()
    root: Path = Path(os.getcwd())


@dataclass
class GitMetadata(FromParams):
    commit: Optional[str] = None
    remote: Optional[str] = None

    @classmethod
    def check_for_repo(cls) -> Optional["GitMetadata"]:
        import subprocess

        try:
            commit = (
                subprocess.check_output("git rev-parse HEAD".split(" ")).decode("ascii").strip()
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
class ExecutorMetadata(FromParams):
    step: str
    platform: PlatformMetadata = PlatformMetadata()
    git: Optional[GitMetadata] = GitMetadata.check_for_repo()
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    duration: Optional[float] = None

    def finalize(self):
        self.finished_at = time.time()
        self.duration = round(self.finished_at - self.started_at, 4)
