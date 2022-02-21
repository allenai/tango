import logging
import multiprocessing as mp
from typing import List, Optional, Set, TypeVar

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
        num_cores: Optional[int] = None,
    ) -> None:
        self.workspace = workspace
        self.include_package = include_package
        self.num_cores = num_cores or mp.cpu_count()

        self.cores_being_used = 0

    def execute_step_graph(self, step_graph: StepGraph):
        """
        Execute a :class:`tango.step_graph.StepGraph`.
        """
        from tango import Step

        # Import included packages to find registered components.
        if self.include_package is not None:
            for package_name in self.include_package:
                import_extra_module(package_name)

        ordered_steps = sorted(step_graph.values(), key=lambda step: step.name)

        # find uncacheable leaf steps
        interior_steps: Set[Step] = set()
        for step in ordered_steps:
            for dependency in step.dependencies:
                interior_steps.add(dependency)
        uncacheable_leaf_steps = {
            step for step in set(step_graph.values()) - interior_steps if not step.cache_results
        }

        # manager = mp.Manager()
        step_processes: List[mp.Process] = []

        count = 0
        while self.has_incomplete_steps(step_graph):
            to_run = self.get_steps_to_run(step_graph)
            print("to_run", to_run)
            if count > 1:
                break
            count += 1
            for step in to_run:
                if self.cores_being_used <= self.num_cores:
                    if step.cache_results:
                        step_process = mp.Process(target=step.ensure_result, args=(self.workspace,))
                    elif step in uncacheable_leaf_steps:
                        step_process = mp.Process(target=step.result, args=(self.workspace,))
                    step_processes.append(step_process)
                    step_process.start()

        # Cannot put this within the loop, because it won't move forward until a level finishes.
        for step_process in step_processes:
            step_process.join()

    def has_incomplete_steps(self, step_graph: StepGraph) -> bool:
        """
        TODO
        """
        for step in step_graph.values():
            if self.workspace.step_info(
                step
            ).state == StepState.INCOMPLETE and not self.failed_dependencies(step):
                return True
        return False

    def get_state(self, step: Step):
        return self.workspace.step_info(step).state

    def failed_dependencies(self, step: Step) -> bool:
        for dependency in step.dependencies:
            if self.get_state(dependency) == StepState.FAILED:
                return True
        return False

    def are_dependencies_available(self, step: Step) -> bool:
        """
        Maybe this should be a `Step` class method?
        """
        for dependency in step.dependencies:
            if self.get_state(dependency) not in [StepState.COMPLETED, StepState.UNCACHEABLE]:
                return False
        return True

    def get_steps_to_run(self, step_graph: StepGraph):
        to_run: Set[Step] = set()
        for step in step_graph.values():
            print(step.name, self.get_state(step))
            if (
                self.are_dependencies_available(step)
                and self.get_state(step) == StepState.INCOMPLETE
            ):
                to_run.add(step)
        return to_run
