import logging
import traceback
from typing import TYPE_CHECKING, List, NamedTuple, Optional, Set, TypeVar

from tango.common.util import import_extra_module
from tango.step_graph import StepGraph
from tango.workspace import Workspace

if TYPE_CHECKING:
    from tango.step import Step

logger = logging.getLogger(__name__)


T = TypeVar("T")


class ExecutorOutput(NamedTuple):
    """
    Describes the outcome of the execution.
    """

    successful: Set[str] = set()
    """Steps which ran successfully."""

    failed: Set[str] = set()
    """Steps that failed."""

    not_run: Set[str] = set()
    """Steps that were not executed (potentially because of failed dependencies."""


class Executor:
    """
    An ``Executor`` is a class that is responsible for running steps and caching their results.
    """

    def __init__(
        self,
        workspace: Workspace,
        include_package: Optional[List[str]] = None,
    ) -> None:
        self.workspace = workspace
        self.include_package = include_package

    def execute_step(self, step: "Step", is_uncacheable_leaf_step: bool = False) -> None:
        # Import included packages to find registered components.
        if self.include_package is not None:
            for package_name in self.include_package:
                import_extra_module(package_name)

        if step.cache_results:
            step.ensure_result(self.workspace)
        elif is_uncacheable_leaf_step:
            step.result(self.workspace)

    def execute_step_graph(
        self, step_graph: StepGraph, run_name: Optional[str] = None
    ) -> ExecutorOutput:
        """
        Execute a :class:`tango.step_graph.StepGraph`. This attempts to execute
        every step in order. If a step fails, its dependent steps are not run,
        but unrelated steps are still executed.
        """

        uncacheable_leaf_steps = step_graph.uncacheable_leaf_steps()
        successful: Set[str] = set()
        failed: Set[str] = set()
        not_run: Set[str] = set()
        error_tracebacks: List[str] = []
        for step in step_graph.values():
            if any([dep.name in failed for dep in step.recursive_dependencies]):
                not_run.add(step.name)
            else:
                try:
                    self.execute_step(step, step in uncacheable_leaf_steps)
                    successful.add(step.name)
                except Exception:
                    failed.add(step.name)
                    error_tracebacks.append(traceback.format_exc())

        for stacktrace in error_tracebacks:
            logger.error(stacktrace)

        return ExecutorOutput(successful=successful, failed=failed, not_run=not_run)
