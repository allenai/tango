import logging
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional, Sequence, TypeVar

from .common.logging import log_exception
from .common.registrable import Registrable
from .common.util import import_extra_module
from .step_graph import StepGraph
from .workspace import Workspace

if TYPE_CHECKING:
    from .step import Step

logger = logging.getLogger(__name__)


T = TypeVar("T")


@dataclass
class ExecutionMetadata:
    logs_location: Optional[str] = None
    """
    Path or URL to the logs for the step's execution.
    """

    result_location: Optional[str] = None
    """
    Path or URL to the result of the step's execution.
    """


@dataclass
class ExecutorOutput:
    """
    Describes the outcome of the execution.
    """

    successful: Dict[str, ExecutionMetadata] = field(default_factory=dict)
    """Steps which ran successfully or were found in the cache."""

    failed: Dict[str, ExecutionMetadata] = field(default_factory=dict)
    """Steps that failed."""

    not_run: Dict[str, ExecutionMetadata] = field(default_factory=dict)
    """Steps that were ignored (usually because of failed dependencies)."""


class Executor(Registrable):
    """
    An ``Executor`` is a class that is responsible for running steps and caching their results.

    This is the base class and default implementation, registered as "default".

    .. note::
        The ``parallelism`` parameter has no effect with this default :class:`Executor`,
        but is part of the API because most subclass implementations allow configuring
        parallelism.
    """

    default_implementation = "default"

    def __init__(
        self,
        workspace: Workspace,
        include_package: Optional[Sequence[str]] = None,
        parallelism: Optional[int] = None,
    ) -> None:
        self.workspace = workspace
        self.include_package = include_package
        self.parallelism = parallelism

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
        Execute a :class:`~tango.step_graph.StepGraph`. This attempts to execute
        every step in order. If a step fails, its dependent steps are not run,
        but unrelated steps are still executed. Step failures will be logged, but
        no exceptions will be raised.
        """
        if self.parallelism is not None:
            warnings.warn(
                "The 'parallelism' parameter has no effect with the default Executor. "
                "If you want to run steps in parallel, consider using the MulticoreExecutor.",
                UserWarning,
            )

        successful: Dict[str, ExecutionMetadata] = {}
        failed: Dict[str, ExecutionMetadata] = {}
        not_run: Dict[str, ExecutionMetadata] = {}
        uncacheable_leaf_steps = step_graph.uncacheable_leaf_steps()

        for step in step_graph.values():
            if not step.cache_results and step not in uncacheable_leaf_steps:
                # If a step is uncacheable and required for another step, it will be
                # executed as part of the downstream step's execution.
                continue
            if any(dep.name in failed for dep in step.recursive_dependencies):
                not_run[step.name] = ExecutionMetadata()
            else:
                try:
                    self.execute_step(step)
                    successful[step.name] = ExecutionMetadata(
                        result_location=self.workspace.step_info(step).result_location
                    )
                except Exception as exc:
                    failed[step.name] = ExecutionMetadata()
                    log_exception(exc, logger)

        return ExecutorOutput(successful=successful, failed=failed, not_run=not_run)

    # NOTE: The reason for having this method instead of just using `execute_step()` to run
    # a single step is that the certain executors, such as the BeakerExecutor, need to
    # serialize steps somehow, and the easiest way to serialize a step is by serializing the
    # whole step config (which can be accessed via the step graph).

    def execute_sub_graph_for_step(
        self, step_graph: StepGraph, step_name: str, run_name: Optional[str] = None
    ) -> ExecutorOutput:
        """
        Execute the sub-graph associated with a particular step in a
        :class:`~tango.step_graph.StepGraph`.
        """
        step = step_graph[step_name]
        try:
            self.execute_step(step)
        except Exception as exc:
            log_exception(exc, logger)
            return ExecutorOutput(failed={step_name: ExecutionMetadata()})
        else:
            return ExecutorOutput(
                successful={
                    step_name: ExecutionMetadata(
                        result_location=self.workspace.step_info(step).result_location
                    )
                }
            )


Executor.register("default")(Executor)
