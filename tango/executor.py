import logging
import traceback
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Sequence, Set, TypeVar

from .common.registrable import Registrable
from .common.util import import_extra_module
from .step_graph import StepGraph
from .workspace import Workspace

if TYPE_CHECKING:
    from .step import Step

logger = logging.getLogger(__name__)


T = TypeVar("T")


@dataclass
class ExecutorOutput:
    """
    Describes the outcome of the execution.
    """

    successful: Set[str]
    """Steps which ran successfully or were found in the cache."""

    failed: Set[str]
    """Steps that failed."""

    not_run: Set[str]
    """Steps that were ignored (usually because of failed dependencies)."""


class Executor(Registrable):
    """
    An ``Executor`` is a class that is responsible for running steps and caching their results.

    This is the base class and default implementation, registered as "default".
    """

    default_implementation = "default"

    def __init__(
        self,
        workspace: Workspace,
        include_package: Optional[Sequence[str]] = None,
    ) -> None:
        self.workspace = workspace
        self.include_package = include_package

    def execute_step(self, step: "Step") -> None:
        # Import included packages to find registered components.
        if self.include_package is not None:
            for package_name in self.include_package:
                import_extra_module(package_name)

        if step.cache_results:
            step.ensure_result(self.workspace)
        else:
            step.result(self.workspace)

    def execute_step_graph(
        self, step_graph: StepGraph, run_name: Optional[str] = None
    ) -> ExecutorOutput:
        """
        Execute a :class:`tango.step_graph.StepGraph`. This attempts to execute
        every step in order. If a step fails, its dependent steps are not run,
        but unrelated steps are still executed. Step failures will be logged, but
        no exceptions will be raised.
        """

        successful: Set[str] = set()
        failed: Set[str] = set()
        not_run: Set[str] = set()
        error_tracebacks: List[str] = []
        uncacheable_leaf_steps = step_graph.uncacheable_leaf_steps()

        for step in step_graph.values():
            if not step.cache_results and step not in uncacheable_leaf_steps:
                # If a step is uncacheable and required for another step, it will be
                # executed as part of the downstream step's execution.
                continue
            if any(dep.name in failed for dep in step.recursive_dependencies):
                not_run.add(step.name)
            else:
                try:
                    self.execute_step(step)
                    successful.add(step.name)
                except Exception:
                    failed.add(step.name)
                    error_tracebacks.append(traceback.format_exc())

        for stacktrace in error_tracebacks:
            logger.error(stacktrace)

        return ExecutorOutput(successful=successful, failed=failed, not_run=not_run)


Executor.register("default")(Executor)
