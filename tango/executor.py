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

        for step in ordered_steps:
            if step.cache_results:
                step.ensure_result(self.workspace)
            elif step in uncacheable_leaf_steps:
                step.result(self.workspace)
