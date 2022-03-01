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
            # TODO: do all Workspace types have self.dir?
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

    def _update_running_steps(self) -> List[str]:
        done = []
        errors = []
        for step_name, process in self._running.items():
            poll_status = process.poll()
            if poll_status is not None:
                logger.debug(f"The process for {step_name} has finished executing.")
                if poll_status == 0:
                    logger.debug(f"The process for {step_name} executed successfully.")
                    done.append(step_name)
                else:
                    logger.debug(f"The process for {step_name} failed during execution.")
                    errors.append(step_name)

        for step_name in done:
            self._running.pop(step_name)

        # TODO: deal with errors. Check StepInfo status update.
        if len(errors) > 0:
            raise RuntimeError("Raising this error for now. Deal with it better.")
        return errors

    def execute_step_graph(self, step_graph: StepGraph):
        """
        Execute a :class:`tango.step_graph.StepGraph`.
        """

        # uncacheable_leaf_steps = step_graph.find_uncacheable_leaf_steps()
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
                self._update_running_steps()
                to_run = self._get_steps_to_run(step_graph)
                logger.debug(f"Steps ready to run: {to_run}")
                for step_name in to_run:
                    self._queue_step(step_name)

                while len(self._queued_steps) > 0 and len(self._running) < self.parallelism:
                    self._try_to_execute_next_step(config_path=file_ref.name)

                import time

                time.sleep(6)
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
        # This hits the sqlite db each time. TODO: confirm that this always guarantees latest updates.
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
            # print(step.name, self._get_state(step))
            if (
                self._are_dependencies_available(step)
                and self._get_state(step) == StepState.INCOMPLETE  # Not already running.
                and step.name not in self._queued_steps
            ):
                to_run.add(step.name)
        return to_run


#
#
# class MulticoreExecutor(Executor):
#     """
#     A ``MulticoreExecutor`` runs the steps in parallel and caches their results.
#     """
#
#     def __init__(
#         self,
#         workspace: Workspace,
#         include_package: Optional[List[str]] = None,
#         num_cores: Optional[int] = None,
#     ) -> None:
#         self.workspace = workspace
#         self.include_package = include_package
#         self.num_cores = num_cores or mp.cpu_count()
#
#         self.cores_being_used = 0
#
#     def execute_step_graph(self, step_graph: StepGraph):
#         """
#         Execute a :class:`tango.step_graph.StepGraph`.
#         """
#         from tango import Step
#
#         # Import included packages to find registered components.
#         if self.include_package is not None:
#             for package_name in self.include_package:
#                 import_extra_module(package_name)
#
#         ordered_steps = sorted(step_graph.values(), key=lambda step: step.name)
#
#         # find uncacheable leaf steps
#         interior_steps: Set[Step] = set()
#         for step in ordered_steps:
#             for dependency in step.dependencies:
#                 interior_steps.add(dependency)
#         uncacheable_leaf_steps = {
#             step for step in set(step_graph.values()) - interior_steps if not step.cache_results
#         }
#
#         # manager = mp.Manager()
#         step_processes: List[mp.Process] = []
#
#         count = 0
#         while self.has_incomplete_steps(step_graph):
#             to_run = self.get_steps_to_run(step_graph)
#             print("to_run", to_run)
#             if count > 1:
#                 break
#             count += 1
#             for step in to_run:
#                 if self.cores_being_used <= self.num_cores:
#                     if step.cache_results:
#                         step_process = mp.Process(target=step.ensure_result, args=(self.workspace,))
#                     elif step in uncacheable_leaf_steps:
#                         step_process = mp.Process(target=step.result, args=(self.workspace,))
#                     step_processes.append(step_process)
#                     step_process.start()
#
#         # Cannot put this within the loop, because it won't move forward until a level finishes.
#         for step_process in step_processes:
#             step_process.join()
#
#     def has_incomplete_steps(self, step_graph: StepGraph) -> bool:
#         """
#         TODO
#         """
#         for step in step_graph.values():
#             if self.workspace.step_info(
#                 step
#             ).state == StepState.INCOMPLETE and not self.failed_dependencies(step):
#                 return True
#         return False
#
#     def get_state(self, step: Step):
#         return self.workspace.step_info(step).state
#
#     def failed_dependencies(self, step: Step) -> bool:
#         for dependency in step.dependencies:
#             if self.get_state(dependency) == StepState.FAILED:
#                 return True
#         return False
#
#     def are_dependencies_available(self, step: Step) -> bool:
#         """
#         Maybe this should be a `Step` class method?
#         """
#         for dependency in step.dependencies:
#             if self.get_state(dependency) not in [StepState.COMPLETED, StepState.UNCACHEABLE]:
#                 return False
#         return True
#
#     def get_steps_to_run(self, step_graph: StepGraph):
#         to_run: Set[Step] = set()
#         for step in step_graph.values():
#             print(step.name, self.get_state(step))
#             if (
#                 self.are_dependencies_available(step)
#                 and self.get_state(step) == StepState.INCOMPLETE
#             ):
#                 to_run.add(step)
#         return to_run
