import logging
from typing import List, Optional, Set, TypeVar

import click

from tango.common.logging import click_logger
from tango.common.util import import_extra_module
from tango.server.workspace_server import WorkspaceServer
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
        server: Optional[WorkspaceServer] = None,
    ) -> None:
        self.workspace = workspace
        self.include_package = include_package
        self.server = server

    def execute_step_graph(self, step_graph: StepGraph) -> str:
        """
        Execute a :class:`tango.step_graph.StepGraph`.
        """
        from tango import Step

        # Import included packages to find registered components.
        if self.include_package is not None:
            for package_name in self.include_package:
                import_extra_module(package_name)

        run_name = self.workspace.register_run(
            step for step in step_graph.values() if step.cache_results
        )
        if self.server is not None:
            click_logger.info(
                "Server started at "
                + click.style(self.server.address_for_display(run_name), bold=True)
            )

        ordered_steps = sorted(step_graph.values(), key=lambda step: step.name)

        # find uncacheable leaf steps
        interior_steps: Set[Step] = set()
        for step in ordered_steps:
            for dependency in step.dependencies:
                interior_steps.add(dependency)
        uncacheable_leaf_steps = {
            step for step in set(step_graph.values()) - interior_steps if not step.cache_results
        }

        click_logger.info(
            click.style("Starting new run ", fg="green")
            + click.style(run_name, fg="green", bold=True)
        )
        for step in ordered_steps:
            if step.cache_results:
                step.ensure_result(self.workspace)
            elif step in uncacheable_leaf_steps:
                step.result(self.workspace)

        # Print everything that has been computed.
        for step in ordered_steps:
            if step in self.workspace.step_cache:
                info = self.workspace.step_info(step)
                click_logger.info(
                    click.style("\N{check mark} The output for ", fg="green")
                    + click.style(f'"{step.name}"', bold=True, fg="green")
                    + click.style(" is in ", fg="green")
                    + click.style(f"{info.result_location}", bold=True, fg="green")
                )

        return run_name
