import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Set, TypeVar, Union
from urllib.parse import ParseResult

import dill
import petname
from sqlitedict import SqliteDict

from tango.common import PathOrStr
from tango.common.file_lock import FileLock
from tango.common.logging import file_handler
from tango.common.util import exception_to_string, utc_now_datetime
from tango.step import Step
from tango.step_cache import StepCache
from tango.step_caches import LocalStepCache
from tango.step_info import StepInfo, StepState
from tango.workspace import Run, StepExecutionMetadata, Workspace

logger = logging.getLogger(__name__)

T = TypeVar("T")


@Workspace.register("local")
class LocalWorkspace(Workspace):
    """
    This is a :class:`.Workspace` that keeps all its data in a local directory. This works great for single-machine
    jobs, or for multiple machines in a cluster if they can all access the same NFS drive.

    :param dir: The directory to store all the data in

    The directory will have three subdirectories, ``cache/`` for the step cache, ``runs/`` for the runs,
    and ``latest/`` for the results of the latest run. For the format of the ``cache/`` directory,
    refer to :class:`.LocalStepCache`. The ``runs/`` directory will contain one subdirectory for each
    registered run. Each one of those contains a symlink from the name of the step to the results directory
    in the step cache. Note that :class:`.LocalWorkspace` creates these symlinks even for steps that have not
    finished yet. You can tell the difference because either the symlink points to a directory that doesn't exist,
    or it points to a directory in the step cache that doesn't contain results.

    .. tip::

        Registered as a :class:`~tango.workspace.Workspace` under the name "local".

        You can also instantiate this workspace from a URL with the scheme ``local://``.
        For example, ``Workspace.from_url("local:///tmp/workspace")`` gives you a :class:`LocalWorkspace`
        in the directory ``/tmp/workspace``.

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
        self.latest_dir = self.dir / "latest"

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
                        stepinfo = dill.load(f)

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

    def __getstate__(self):
        """
        We override `__getstate__()` to customize how instances of this class are pickled
        since we don't want to persist certain attributes.
        """
        out = super().__getstate__()
        out["locks"] = {}
        return out

    @property
    def url(self) -> str:
        return "local://" + str(self.dir)

    @classmethod
    def from_parsed_url(cls, parsed_url: ParseResult) -> "Workspace":
        workspace_dir: Path
        if parsed_url.netloc:
            workspace_dir = Path(parsed_url.netloc)
            if parsed_url.path:
                workspace_dir = workspace_dir / parsed_url.path.lstrip("/")
        elif parsed_url.path:
            workspace_dir = Path(parsed_url.path)
        else:
            workspace_dir = Path(".")
        return cls(workspace_dir.resolve())

    def step_dir(self, step_or_unique_id: Union[Step, str]) -> Path:
        return self.cache.step_dir(step_or_unique_id)

    @property
    def step_cache(self) -> StepCache:
        return self.cache

    def work_dir(self, step: Step) -> Path:
        result = self.step_dir(step) / "work"
        result.mkdir(parents=True, exist_ok=True)
        return result

    @classmethod
    def guess_step_dir_state(cls, dir: Path) -> Set[StepState]:
        """
        Returns the possible states of a given step dir, to the best of our knowledge.

        :param dir: the step dir to example
        :return: a set of possible states for the step
        """

        # If the directory doesn't exist, the step is incomplete or uncacheable.
        if not dir.exists():
            return {StepState.INCOMPLETE, StepState.UNCACHEABLE}

        # If the lock file exists and is locked, the step is running.
        lock_file = dir / "lock"
        if lock_file.exists():
            lock = FileLock(lock_file)
            try:
                lock.acquire(0.1)
                lock.release()
            except TimeoutError:
                return {StepState.RUNNING}

        # If the directory is empty except for the work dir and the lock file, the step is running, incomplete,
        # or failed. But it can't be running because then the lockfile would be locked, so it can only be
        # incomplete or failed.
        for dir_entry in dir.iterdir():
            if dir_entry.name == "work" and dir_entry.is_dir():
                continue
            if dir_entry.name == "lock" and dir_entry.is_file():
                continue
            break
        else:
            return {StepState.INCOMPLETE, StepState.FAILED}

        return set(StepState)

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

                    step_info = StepInfo.new_from_step(step)
                    d[unique_id] = step_info
                    del step

                # Perform some sanity checks. Sqlite and the file system can get out of sync
                # when a process dies suddenly.
                step_dir = self.step_dir(unique_id)
                step_state_guesses = self.guess_step_dir_state(step_dir) or step_info.state
                if step_info.state not in step_state_guesses:
                    if step_info.state == StepState.RUNNING:
                        # We think the step is running, but it can't possibly be running, so we go ahead and
                        # assume the step is incomplete.
                        step_info.start_time = None
                        step_info.end_time = None
                        d[unique_id] = step_info
                    else:
                        possible_states = ", ".join(s.value for s in step_state_guesses)
                        raise IOError(
                            f"The step '{unique_id}' is marked as being {step_info.state.value}, but we "
                            f"determined it can only be one of {{{possible_states}}}. If you are positive "
                            f"this is a screw-up, delete the directory at '{step_dir}' and try again."
                        )

                return step_info

            result = find_or_add_step_info(step_or_unique_id)
            d.commit()
            self._fix_step_info(result)
            return result

    def _step_lock_file(self, step_or_unique_id: Union[Step, str]) -> Path:
        step_dir = self.step_dir(step_or_unique_id)
        step_dir.mkdir(parents=True, exist_ok=True)
        return step_dir / "lock"

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
            step_info.start_time = utc_now_datetime()
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
            metadata = StepExecutionMetadata(
                step=step.unique_id, config=replace_steps_with_unique_id(config)
            )
            # Finalize metadata and save to run directory.
            metadata.save(self.step_dir(step))

        # Mark the step as finished
        step_info.end_time = utc_now_datetime()
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
            step_info.end_time = utc_now_datetime()
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

        self._save_registered_run(name, all_steps)

        # write targets
        for target in targets:
            if target.cache_results:
                target_path = self.step_dir(target)
                (run_dir / target.name).symlink_to(os.path.relpath(target_path, run_dir))

        # Note: Python3.7 pathlib.Path.unlink does not support the `missing_ok` argument.
        if self.latest_dir.is_symlink():
            self.latest_dir.unlink()
        self.latest_dir.symlink_to(run_dir)

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
        steps_for_run = self._load_registered_run(name)
        return Run(name, steps_for_run, datetime.fromtimestamp(run_dir.stat().st_ctime))

    def _run_step_info_file(self, name: str) -> Path:
        return self.runs_dir / name / "stepinfo.json"

    def _save_registered_run(self, name: str, all_steps: Iterable[Step]) -> None:
        step_unique_ids = {}
        with SqliteDict(self.step_info_file) as d:
            for step in all_steps:
                try:
                    step_info = d[step.unique_id]
                    step_info.name = step.name
                    d[step.unique_id] = step_info
                except KeyError:
                    d[step.unique_id] = StepInfo.new_from_step(step)
                step_unique_ids[step.name] = step.unique_id

            d.commit()

            run_step_info_file = self._run_step_info_file(name)
            with open(run_step_info_file, "w") as file_ref:
                json.dump(step_unique_ids, file_ref)

    def _load_registered_run(self, name: str) -> Dict[str, StepInfo]:
        run_step_info_file = self._run_step_info_file(name)
        try:
            with open(run_step_info_file, "r") as file_ref:
                step_ids = json.load(file_ref)
        except FileNotFoundError:
            # for backwards compatibility
            run_dir = self.runs_dir / name
            step_ids = {}
            for step_symlink in run_dir.iterdir():
                if not step_symlink.is_symlink():
                    continue
                step_name = str(step_symlink.name)
                unique_id = str(step_symlink.resolve().name)
                step_ids[step_name] = unique_id

        with SqliteDict(self.step_info_file, flag="r") as d:
            steps_for_run = {}
            for step_name, unique_id in step_ids.items():
                step_info = d[unique_id]
                assert isinstance(step_info, StepInfo)
                self._fix_step_info(step_info)
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

    def capture_logs_for_run(self, name: str):
        return file_handler(self.run_dir(name) / "out.log")
