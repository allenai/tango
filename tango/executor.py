import logging
from typing import List, Optional

import click

from tango.common.util import import_extra_module
from tango.step import Step
from tango.step_graph import StepGraph
from tango.workspace import Workspace

logger = logging.getLogger(__name__)


class Executor:
    """
    An ``Executor`` is a class that is responsible for running steps and caching their results.
    """

    def __init__(self, workspace: Workspace, include_package: Optional[List[str]] = None) -> None:
        self.workspace = workspace
        self.include_package = include_package

    def execute_step_graph(self, step_graph: StepGraph) -> None:
        """
        Execute an entire :class:`tango.step_graph.StepGraph`.
        """
        # Import included packages to find registered components.
        if self.include_package is not None:
            for package_name in self.include_package:
                import_extra_module(package_name)

        self.workspace.register_run(step_graph.values())

        for step in step_graph.values():
            if step.cache_results:
                self.execute_step(step)

        # Print everything that has been computed.
        for step in step_graph.values():
            if step in self.workspace.step_cache:
                info = self.workspace.step_info(step)
                click.echo(
                    click.style("\N{check mark} The output for ", fg="green")
                    + click.style(f'"{step.name}"', bold=True, fg="green")
                    + click.style(" is in ", fg="green")
                    + click.style(f"{info.result_location}", bold=True, fg="green")
                )

    def execute_step(self, step: Step, quiet: bool = False) -> None:
        if not quiet:
            click.echo(
                click.style("\N{black circle} Starting run for ", fg="blue")
                + click.style(f'"{step.name}"...', bold=True, fg="blue")
            )

        # Run the step.
        step.ensure_result(self.workspace)

        if not quiet:
            click.echo(
                click.style("\N{check mark} Finished run for ", fg="green")
                + click.style(f'"{step.name}"', bold=True, fg="green")
            )
