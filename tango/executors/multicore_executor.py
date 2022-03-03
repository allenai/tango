import json
import logging
import os
import subprocess
from typing import Dict, List, Optional, OrderedDict, Set, TypeVar

from tango.common.util import import_extra_module
from tango.executors.executor import Executor
from tango.step import Step
from tango.step_graph import StepGraph
from tango.workspace import StepState, Workspace

logger = logging.getLogger(__name__)

T = TypeVar("T")


class MulticoreExecutor(Executor):
    """
    A ``MulticoreExecutor`` runs the steps in parallel and caches their results.
    """

    def __init__(
        self,
        workspace: Workspace,
        include_package: Optional[List[str]] = None,
        parallelism: int = 1,
        num_tries_to_sync_states: int = 3,
        wait_seconds_to_sync_states: int = 3,
    ) -> None:
        self.workspace = workspace
        self.include_package = include_package
        self.parallelism = parallelism

        self._num_tries_to_sync_states = num_tries_to_sync_states
        self._wait_seconds_to_sync_states = wait_seconds_to_sync_states

        self._reset_execution()

    def _reset_execution(self):
        self._running: OrderedDict[str, subprocess.Popen] = OrderedDict({})
        # self._running: Set[str] = set()
        # # We save the steps that have completed/failed so that we only query step states once
        # # per loop.
        # self._done: Set[str] = set()
        self._failed: Set[str] = set()
        self._queued_steps: List[str] = []  # TODO: do we need a fancier Queue object?

    def _queue_step(self, step_name: str) -> None:
        self._queued_steps.append(step_name)
        logger.debug(f"Step {step_name} added to the queue for execution.")

    def execute_step(self, step, is_uncacheable_leaf_step: bool = False):
        # TODO: this is already implemented in base class Executor. Remove after debugging.
        # Note: did not add type information because of circular imports.

        # Import included packages to find registered components.
        if self.include_package is not None:
            for package_name in self.include_package:
                import_extra_module(package_name)
        logger.debug(f"Beginning execution for {step.name}")
        if step.cache_results:
            step.ensure_result(self.workspace)
        elif is_uncacheable_leaf_step:
            step.result(self.workspace)
        logger.debug(f"Step {step.name} finished executing.")

    def _try_to_execute_next_step(
        self,
        config_path: str,
        overrides: Optional[Dict] = None,
    ) -> None:
        if len(self._queued_steps) == 0:
            logger.debug("No steps in queue!")
            return
        if len(self._running) < self.parallelism:
            step_name = self._queued_steps.pop(0)
            # TODO: also check the StepState here for sanity, in case there are 2 concurrent experiment runs.
            # TODO: will all Workspace types have self.dir?
            command = f"tango run {config_path} -w {self.workspace.dir} -s {step_name} --no-server"  # type: ignore
            if self.include_package is not None:
                for package in self.include_package:
                    command += f" -i {package}"
            if overrides is not None:
                command += f" -o {json.dumps(overrides)}"
            process = subprocess.Popen(command, shell=True)
            self._running[step_name] = process
        else:
            logger.debug(
                f"{self.parallelism} steps are already running. Will attempt to execute later."
            )

    def _update_running_steps(self, step_states: Dict[str, StepState]) -> List[str]:
        done = []
        errors = []
        for step_name, process in self._running.items():
            poll_status = process.poll()
            if poll_status is not None:
                # We use poll_status to check if the process ended, but the StepState for checking
                # completion/failure status because after the process ends, the lock release etc. takes a while.
                if step_states[step_name] == StepState.COMPLETED:
                    done.append(step_name)
                elif (
                    step_states[step_name] == StepState.FAILED
                    or step_states[step_name] == StepState.INCOMPLETE
                ):
                    # TODO: look into why the step status changes from running back to incomplete sometimes.
                    errors.append(step_name)
                else:
                    pass

        # print("errors:", errors)
        for step_name in done + errors:
            self._running.pop(step_name)

        for step_name in errors:
            self._failed.add(step_name)

        # # # TODO: deal with errors.
        # if len(errors) > 0:
        #     raise RuntimeError("Raising this error for now. Deal with it better.")
        return errors

    def _sync_step_states(self, step_graph: StepGraph) -> Dict[str, StepState]:
        # update the step_states.
        # Although, this is not really elegant. The issue is: main multicore executor run queues a step,
        # it loads up the config again, and begins execution in a different process. If we do this check
        # before it has had time to update the StepState, it'll throw the out of sync error in
        # LocalWorkspace.
        import time

        attempts = 0
        while attempts < self._num_tries_to_sync_states:
            attempts += 1
            try:
                step_states = {step.name: self._get_state(step) for step in step_graph.values()}
                break
            except IOError:
                step_states = {}
                time.sleep(self._wait_seconds_to_sync_states)
        return step_states

    def execute_step_graph(self, step_graph: StepGraph):
        """
        Execute a :class:`tango.step_graph.StepGraph`.
        """

        self._reset_execution()

        from tempfile import NamedTemporaryFile

        # TODO: use Tango global settings cache as "dir".
        with NamedTemporaryFile(prefix="step-graph-to-file-run", suffix=".jsonnet") as file_ref:
            step_graph.to_file(file_ref.name)
            assert os.path.exists(file_ref.name)

            step_states = self._sync_step_states(step_graph)
            # count = 7
            while self._has_incomplete_steps(step_graph, step_states):
                self._update_running_steps(step_states)
                to_run = self._get_steps_to_run(step_graph, step_states)
                if to_run:
                    logger.debug(f"Steps ready to run: {to_run}")
                for step_name in to_run:
                    self._queue_step(step_name)

                while len(self._queued_steps) > 0 and len(self._running) < self.parallelism:
                    self._try_to_execute_next_step(
                        config_path=file_ref.name
                    )  # maybe use StepInfo right before starting run.

                step_states = self._sync_step_states(step_graph)
                if not step_states:
                    # TODO: better message.
                    raise RuntimeError("A reasonable error message.")

                # import time
                # time.sleep(0)

                # if not self._running:
                #     count -= 1
                # if count <=0:
                #     print(step_states)
                #     print(self._running)
                #     print(self._queued_steps)
                #     raise RuntimeError("Breaking loop!")

    def _has_incomplete_steps(
        self, step_graph: StepGraph, step_states: Dict[str, StepState]
    ) -> bool:
        """
        TODO
        """

        def _failed_dependencies(step: Step) -> bool:
            for dependency in step.dependencies:
                if (
                    step_states[dependency.name] == StepState.FAILED
                    or dependency.name in self._failed
                ):
                    return True
            return False

        print("Running: ", self._running)
        print("Queued: ", self._queued_steps)
        print("Step states: ", step_states)
        for step_name, step_state in step_states.items():
            if (
                step_name in self._running
                or step_name in self._queued_steps
                or (
                    step_state in [StepState.INCOMPLETE, StepState.RUNNING]
                    and not _failed_dependencies(step_graph[step_name])
                    and step_name not in self._failed
                )
            ):
                return True
        return False

    def _get_state(self, step: Step):
        # Note: This hits the sqlite db each time.
        return self.workspace.step_info(step).state

    def _get_steps_to_run(
        self, step_graph: StepGraph, step_states: Dict[str, StepState]
    ) -> Set[str]:
        def _are_dependencies_available(step: Step) -> bool:
            for dependency in step.dependencies:
                if step_states[dependency.name] not in [StepState.COMPLETED, StepState.UNCACHEABLE]:
                    return False
            return True

        to_run: Set[str] = set()
        for step in step_graph.values():
            if (
                _are_dependencies_available(step)
                and step.name not in self._running  # Not already running.
                and step.name not in self._queued_steps  # Not queued to run.
                and step.name not in self._failed  # Not already failed.
                and step_states[step.name] == StepState.INCOMPLETE
            ):
                to_run.add(step.name)
        return to_run
