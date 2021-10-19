import logging
import os
from typing import Optional, Union, List, Sequence

import click
from click_help_colors import HelpColorsCommand, HelpColorsGroup

from tango.version import VERSION


@click.group(
    cls=HelpColorsGroup,
    help_options_color="green",
    help_headers_color="yellow",
    context_settings={"max_content_width": 115},
)
@click.version_option(version=VERSION)
@click.option(
    "--log-level",
    help="Set the global log level.",
    type=click.Choice(["debug", "info", "warning", "error"], case_sensitive=False),
    show_choices=True,
    default="info",
)
@click.option(
    "--no-logging",
    is_flag=True,
    help="Disable logging altogether.",
)
def main(log_level, no_logging):
    if not no_logging:
        level = logging._nameToLevel[log_level.upper()]
        logging.basicConfig(
            format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
            level=level,
        )
        # filelock emits too many messages, so tell it to be quiet unless it has something
        # important to say.
        logging.getLogger("filelock").setLevel(max(level, logging.WARNING))

    from tango.common.util import install_sigterm_handler

    # We want to be able to catch SIGTERM signals in addition to SIGINT (keyboard interrupt).
    install_sigterm_handler()


@main.command(
    cls=HelpColorsCommand,
    help_options_color="green",
    help_headers_color="yellow",
    context_settings={"max_content_width": 115},
)
@click.argument(
    "experiment",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
)
@click.option(
    "-d",
    "--directory",
    type=click.Path(file_okay=False),
    help="""The directory in which to save the results of each step. If not specified,
    a named temporary directory will be created.""",
    default=None,
)
@click.option(
    "-o",
    "--overrides",
    type=str,
    help="""A JSON(NET) string used to override fields in the experiment config.
    Use dot syntax to specify nested fields.""",
)
@click.option(
    "-i",
    "--include-package",
    type=str,
    help="Python packages or modules to import for tango components.",
    multiple=True,
)
@click.option(
    "--file-friendly-logging",
    is_flag=True,
    help="Outputs progress bar status on separate lines and slows refresh rate",
)
def run(
    experiment: str,
    directory: Optional[Union[str, os.PathLike]] = None,
    overrides: Optional[str] = None,
    include_package: Optional[Sequence[str]] = None,
    file_friendly_logging: bool = False,
):
    """
    Run a tango experiment.

    EXPERIMENT is the path to experiment's JSON/Jsonnet/YAML configuration file.
    """
    _run(
        experiment,
        directory=directory,
        overrides=overrides,
        include_package=include_package,
        file_friendly_logging=file_friendly_logging,
    )


@main.command(
    cls=HelpColorsCommand,
    help_options_color="green",
    help_headers_color="yellow",
    context_settings={"max_content_width": 115},
)
def info():
    import platform

    from tango.common.util import find_integrations, import_module_and_submodules

    click.echo(f"Tango version {VERSION} (python {platform.python_version()})\n")
    click.echo("Integrations:")
    for integration in find_integrations():
        name = integration.split(".")[-1]
        is_installed = True
        try:
            import_module_and_submodules(integration)
        except (ModuleNotFoundError, ImportError):
            is_installed = False
        if is_installed:
            click.secho(f" ✓ {name}", fg="green")
        else:
            click.secho(f" ✗ {name}", fg="yellow")


def _run(
    experiment: str,
    directory: Optional[Union[str, os.PathLike]] = None,
    overrides: Optional[str] = None,
    include_package: Optional[Sequence[str]] = None,
    file_friendly_logging: bool = False,
):
    if file_friendly_logging:
        os.environ["FILE_FRIENDLY_LOGGING"] = "true"

    from pathlib import Path
    from tango.common.params import Params
    from tango.common.util import import_module_and_submodules
    from tango.executor import Executor
    from tango.step_cache import StepCache
    from tango.step_graph import StepGraph

    # Read params.
    params = Params.from_file(experiment, params_overrides=overrides or "")

    # Import included packages to find registered components.
    # NOTE: The Executor imports these as well because it's meant to be used
    # directly, but we also need to import here in case the user is using a
    # custom Executor or StepCache.
    include_package: List[str] = list(include_package or [])
    include_package += params.pop("include_package", [])
    for package_name in include_package:
        import_module_and_submodules(package_name)

    # Prepare directory.
    if directory is None:
        from tempfile import mkdtemp

        directory = mkdtemp(prefix="tango-")
        click.echo(
            "Creating temporary directory for run: " + click.style(f"{directory}", fg="yellow")
        )
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    # Initialize step graph, cache, and executor.
    step_graph = StepGraph(params.pop("steps", keep_as_dict=True))
    step_cache = StepCache.from_params(
        params.pop("cache", default={}), dir=directory / "step_cache"
    )
    executor = Executor.from_params(
        params.pop("executor", default={}),
        dir=directory,
        step_cache=step_cache,
        include_package=include_package,
    )

    # Now executor the step graph.
    executor.execute_step_graph(step_graph)


if __name__ == "__main__":
    main()
