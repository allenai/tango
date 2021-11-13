import logging
from typing import List, Optional, TypeVar

import click

from tango.common.util import import_extra_module
from tango.step import Step
from tango.step_graph import StepGraph
from tango.workspace import Workspace

logger = logging.getLogger(__name__)


T = TypeVar("T")


class Executor:
    """
    An ``Executor`` is a class that is responsible for running steps and caching their results.
    """

    def __init__(self, workspace: Workspace, include_package: Optional[List[str]] = None) -> None:
        self.workspace = workspace
        self.include_package = include_package

    def execute_step_graph(self, step_graph: StepGraph) -> str:
        """
        Execute a :class:`tango.step_graph.StepGraph`.
        """
        # Import included packages to find registered components.
        if self.include_package is not None:
            for package_name in self.include_package:
                import_extra_module(package_name)

        run_name = self.workspace.register_run(step_graph.values())

        ordered_steps = sorted(step_graph.values(), key=lambda step: step.name)

            # Execute all steps that need to run, i.e. steps that fall into one of the
            # following two categories:
            #  1. step should be cached but is not in cache
            #  2. step is a dependency (direct or recursively) to another step that should be cached
            #     but is not in the cache.
            for step in ordered_steps:
            if step.cache_results:
                self.execute_step_with_dependencies(step)

        # Print everything that has been computed.
        for step in ordered_steps:
            if step in self.workspace.step_cache:
                info = self.workspace.step_info(step)
                click.echo(
                    click.style("\N{check mark} The output for ", fg="green")
                    + click.style(f'"{step.name}"', bold=True, fg="green")
                    + click.style(" is in ", fg="green")
                    + click.style(f"{info.result_location}", bold=True, fg="green")
                )

        return run_name

    def execute_step(self, step: Step, quiet: bool = False) -> None:
        if not quiet:
            click.echo(
                click.style("\N{black circle} Starting run for ", fg="blue")
                + click.style(f'"{step.name}"', bold=True, fg="blue")
                + click.style(
                    "..." if needed_by is None else f' (needed by "{needed_by.name}")...', fg="blue"
                )
            )

        # Run the step.
        step.ensure_result(self.workspace)

        if not quiet:
            click.echo(
                click.style("\N{check mark} Finished run for ", fg="green")
                + click.style(f'"{step.name}"', bold=True, fg="green")
            )
