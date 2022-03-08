import logging
import os
import subprocess
from typing import Dict, List, Optional, OrderedDict, Set, TypeVar

from tango.executor import Executor
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

        # Perhaps there's a better way to do this without these being passed as args.
        self._num_tries_to_sync_states = num_tries_to_sync_states
        self._wait_seconds_to_sync_states = wait_seconds_to_sync_states

        self._reset_execution()

    def _reset_execution(self) -> None:
        self._running: OrderedDict[str, subprocess.Popen] = OrderedDict({})
        self._failed: Set[str] = set()
        self._queued_steps: List[str] = []

    def _queue_step(self, step_name: str) -> None:
        self._queued_steps.append(step_name)
        logger.debug(f"Step {step_name} added to the queue for execution.")

    def _try_to_execute_next_step(
        self,
        config_path: str,
        run_name: Optional[str] = None,
    ) -> None:
        """
        If there are queued steps, try to start processes for them (limited by `parallelism`).
        """
        if len(self._queued_steps) == 0:
            logger.debug("No steps in queue!")
            return
        if len(self._running) < self.parallelism:
            step_name = self._queued_steps.pop(0)
            # TODO: also check the StepState here for sanity, in case there are 2 concurrent experiment runs?
            command = f"tango run {config_path} -s {step_name} --no-server"
            if hasattr(self.workspace, "dir"):
                command += f" -w {self.workspace.dir}"  # type: ignore
            if self.include_package is not None:
                for package in self.include_package:
                    command += f" -i {package}"
            if run_name is not None:
                command += f" -n {run_name}"
            process = subprocess.Popen(command, shell=True)
            self._running[step_name] = process
        else:
            logger.debug(
                f"{self.parallelism} steps are already running. Will attempt to execute later."
            )

    def _update_running_steps(self, step_states: Dict[str, StepState]) -> List[str]:
        """
        Check the running processes for status. We use poll_status to check if the process ended,
        but the StepState for checking completion/failure status, because after the process ends,
        the lock release etc. sometimes takes a beat longer.
        """
        done = []
        errors = []
        for step_name, process in self._running.items():
            poll_status = process.poll()
            if poll_status is not None:
                if step_states[step_name] == StepState.COMPLETED:
                    done.append(step_name)
                elif (
                    step_states[step_name] == StepState.FAILED
                    or step_states[step_name] == StepState.INCOMPLETE
                ):
                    # TODO: look into why the step status changes from running back to incomplete sometimes.
                    # Possibly it's due to the workspace being aggressive in marking it as incomplete when
                    # it thinks that the process is not running.
                    errors.append(step_name)
                else:
                    pass

        for step_name in done + errors:
            self._running.pop(step_name)

        for step_name in errors:
            self._failed.add(step_name)

        return errors

    def _sync_step_states(self, step_graph: StepGraph) -> Dict[str, StepState]:
        """
        Update the StepState info.
        Although, this is not really elegant. The issue is as follows: The main multicore executor process
        queues a step, and starts step execution in a different process. If we try to read the StepState
        before that process has had time to update the StepState, the Workspace will throw the out of sync
        error (IOError: process should be running but it's considered incomplete...).

        Hence, we try to read a few times, so that the child process has time to update the step's state.
        """

        import time

        attempts = 0
        while attempts < self._num_tries_to_sync_states:
            attempts += 1
            try:
                step_states = {step.name: self._get_state(step) for step in step_graph.values()}
                break
            except IOError:
                if attempts == self._num_tries_to_sync_states:
                    raise
                step_states = {}
                time.sleep(self._wait_seconds_to_sync_states)
        return step_states

    def execute_step_graph(self, step_graph: StepGraph, run_name: Optional[str] = None):
        """
        Execute a :class:`tango.step_graph.StepGraph`.
        """

        self._reset_execution()

        from tempfile import NamedTemporaryFile

        # Creates a temporary file in which to store the config. This is passed as a command line
        # argument to child step processes.
        with NamedTemporaryFile(prefix="step-graph-to-file-run", suffix=".jsonnet") as file_ref:
            step_graph.to_file(file_ref.name)
            assert os.path.exists(file_ref.name)

            step_states = self._sync_step_states(step_graph)

            while self._has_incomplete_steps(step_graph, step_states):
                # Cleanup previously running steps.
                self._update_running_steps(step_states)

                # Get steps that are ready to run.
                to_run = self._get_steps_to_run(step_graph, step_states)
                if to_run:
                    logger.debug(f"Steps ready to run: {to_run}")

                for step_name in to_run:
                    self._queue_step(step_name)

                # Begin processes for any queued steps (if not enough processes are already running).
                while len(self._queued_steps) > 0 and len(self._running) < self.parallelism:
                    self._try_to_execute_next_step(config_path=file_ref.name, run_name=run_name)

                # Re-sync the StepState info.
                step_states = self._sync_step_states(step_graph)

        assert not self._running and not self._queued_steps

    @property
    def failed_steps(self) -> Set[str]:
        return self._failed

    def _has_incomplete_steps(
        self, step_graph: StepGraph, step_states: Dict[str, StepState]
    ) -> bool:
        """
        Are there any steps in the graph that are currently:
        1) running, or
        2) queued, or
        3) incomplete (with no failed dependencies).

        If there are any failed dependencies for a step, it will never manage to run.
        """

        def _failed_dependencies(step: Step) -> bool:
            for dependency in step.dependencies:
                if (
                    step_states[dependency.name] == StepState.FAILED
                    or dependency.name in self._failed
                ):
                    return True
            return False

        for step_name, step_state in step_states.items():
            if (
                step_name in self._running
                or step_name in self._queued_steps
                or (
                    # If the workspace already has a previous run, we disregard the failure.
                    step_state in [StepState.INCOMPLETE, StepState.FAILED]
                    and not _failed_dependencies(step_graph[step_name])
                    # We check for failures in this run.
                    and step_name not in self._failed
                )
            ):
                return True
        return False

    def _get_state(self, step: Step) -> StepState:
        """
        Returns the StepState as determined by the workspace.
        """
        # Note: This hits the sqlite db each time.
        return self.workspace.step_info(step).state

    def _get_steps_to_run(
        self, step_graph: StepGraph, step_states: Dict[str, StepState]
    ) -> Set[str]:
        """
        Returns the steps that can be queued to run immediately.
        Criteria:
            1) All dependencies are available.
            2) Step is not already running or queued.
            3) Step has not run in the past and failed.
            4) Step's state is INCOMPLETE

        (We only run uncacheable steps if they are needed for another step downstream,
        as part of the downstream step).
        """

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
                # See comment in _has_incomplete_steps
                and step_states[step.name] in [StepState.INCOMPLETE, StepState.FAILED]
            ):
                to_run.add(step.name)
        return to_run
