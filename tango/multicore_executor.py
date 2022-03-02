import json
import logging
import os
import subprocess
from typing import Dict, List, Optional, OrderedDict, Set, TypeVar

from tango.common.util import import_extra_module
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
    ) -> None:
        self.workspace = workspace
        self.include_package = include_package
        self.parallelism = parallelism

        self._running: OrderedDict[str, subprocess.Popen] = OrderedDict({})
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

    def _update_running_steps(self, step_graph: StepGraph) -> List[str]:
        done = []
        errors = []
        for step_name, process in self._running.items():
            poll_status = process.poll()
            if poll_status is not None:
                logger.debug(f"The process for {step_name} has finished executing.")
                # TODO: also check the step_info state. Only check that?
                if (
                    poll_status == 0
                    and self._get_state(step_graph[step_name]) == StepState.COMPLETED
                ):
                    logger.debug(f"The process for {step_name} executed successfully.")
                    done.append(step_name)
                elif self._get_state(step_graph[step_name]) == StepState.FAILED:
                    logger.debug(f"The process for {step_name} failed during execution.")
                    errors.append(step_name)
                else:
                    # TODO: better logs.
                    # Ideally this would be the state when the process has ended, but cleanup
                    # is happening; i.e., releasing locks etc.
                    pass

        for step_name in done:
            self._running.pop(step_name)

        for step_name in errors:
            self._failed.add(step_name)

        # TODO: deal with errors. Check StepInfo status update.
        if len(errors) > 0:
            raise RuntimeError("Raising this error for now. Deal with it better.")
        return errors

    def execute_step_graph(self, step_graph: StepGraph):
        """
        Execute a :class:`tango.step_graph.StepGraph`.
        """

        from tempfile import NamedTemporaryFile

        # TODO: use Tango global settings cache as "dir".
        file_ref = NamedTemporaryFile(
            prefix="step-graph-to-file-run", suffix=".jsonnet", delete=False
        )
        step_graph.to_file(file_ref.name)
        assert os.path.exists(file_ref.name)

        try:
            count = 7
            while self._has_incomplete_steps(step_graph):
                self._update_running_steps(step_graph)
                to_run = self._get_steps_to_run(step_graph)
                logger.debug(f"Steps ready to run: {to_run}")
                for step_name in to_run:
                    self._queue_step(step_name)

                while len(self._queued_steps) > 0 and len(self._running) < self.parallelism:
                    self._try_to_execute_next_step(config_path=file_ref.name)

                import time

                time.sleep(5)
                # For debugging.
                count -= 1
                if count <= 0:
                    logger.debug("Coming out of the while loop because count exceeded.")
                    break
        finally:
            os.remove(file_ref.name)

    def _has_incomplete_steps(self, step_graph: StepGraph) -> bool:
        """
        TODO
        """
        for step in step_graph.values():
            if self.workspace.step_info(
                step
            ).state == StepState.INCOMPLETE and not self._failed_dependencies(step):
                return True
        return False

    def _get_state(self, step: Step):
        # Note: This hits the sqlite db each time.
        return self.workspace.step_info(step).state

    def _failed_dependencies(self, step: Step) -> bool:
        for dependency in step.dependencies:
            if self._get_state(dependency) == StepState.FAILED:
                return True
        return False

    def _are_dependencies_available(self, step: Step) -> bool:
        """
        TODO: Maybe this should be a `Step` class method?
        """
        for dependency in step.dependencies:
            if self._get_state(dependency) not in [StepState.COMPLETED, StepState.UNCACHEABLE]:
                return False
        return True

    def _get_steps_to_run(self, step_graph: StepGraph) -> Set[str]:
        to_run: Set[str] = set()
        for step in step_graph.values():
            if (
                self._are_dependencies_available(step)
                and self._get_state(step) == StepState.INCOMPLETE  # Not already running.
                and step.name not in self._queued_steps  # Not queued to run.
            ):
                to_run.add(step.name)
        return to_run
