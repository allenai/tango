import logging
from typing import List, Optional, Set, TypeVar

from tango.common.util import import_extra_module
from tango.step_graph import StepGraph
from tango.workspace import Workspace

logger = logging.getLogger(__name__)


T = TypeVar("T")


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

    def execute_step(self, step, is_uncacheable_leaf_step=False):
        # Note: did not add type information because of circular imports.

        # Import included packages to find registered components.
        if self.include_package is not None:
            for package_name in self.include_package:
                import_extra_module(package_name)

        if step.cache_results:
            step.ensure_result(self.workspace)
        elif is_uncacheable_leaf_step:
            step.result(self.workspace)

    def execute_step_graph(self, step_graph: StepGraph, run_name: Optional[str] = None):
        """
        Execute a :class:`tango.step_graph.StepGraph`.
        """

        ordered_steps = sorted(step_graph.values(), key=lambda step: step.name)
        uncacheable_leaf_steps = step_graph.find_uncacheable_leaf_steps()

        for step in ordered_steps:
            self.execute_step(step, step in uncacheable_leaf_steps)

    @property
    def failed_steps(self) -> Set[str]:
        return set()
