"""
The Tango CLI is the recommended tool to run experiments with.
It also comes with several other useful commands.

You can see the the list of all available commands by running:

.. code-block::

    $ tango --help

.. testcode::
    :hide:

    import subprocess
    output = subprocess.run("tango --help".split(" "), capture_output=True)
    output.check_returncode()
    print(output.stdout.decode().replace("\\n\\n", "\\n").strip())

.. testoutput::

    Usage: tango [OPTIONS] COMMAND [ARGS]...

    Options:
      --version                       Show the version and exit.
      --config FILE                   Path to a global tango.yml settings file.
      --log-level [debug|info|warning|error]
                                      Set the global log level.
      --no-logging                    Disable logging altogether.
      --file-friendly-logging         Outputs progress bar status on separate lines and slows refresh rate.
      --help                          Show this message and exit.

    Commands:
      info  Get info about the current tango installation.
      run   Run a tango experiment.

To see all of the available arguments and options for a particular command, run

.. code-block::

    $ tango [COMMAND] --help

For example,

.. code-block::

    $ tango run --help

``tango run``
-------------

The ``run`` command is used to execute a tango experiment from an experiment configuration file.
See the `Configuration files </overview.html#configuration-files>`_ section in the overview
for a quick introduction to the format.

``tango info``
--------------

The ``info`` command just prints out some useful information about the current tango installation,
such as which integrations are available.

"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

import click
from click_help_colors import HelpColorsCommand, HelpColorsGroup

import tango.common.logging as common_logging
from tango.common.aliases import PathOrStr
from tango.common.from_params import FromParams
from tango.common.params import Params
from tango.version import VERSION


@dataclass
class TangoGlobalSettings(FromParams):
    """
    Defines global settings for tango.
    """

    include_package: Optional[List[str]] = None
    """
    An list of modules where custom registered steps or classes can be found.
    """

    no_logging: bool = False
    """
    If ``True``, logging is disabled.
    """

    log_level: Optional[str] = "info"
    """
    The log level to use. Options are "debug", "info", "warning", and "error".
    """

    _path: Optional[Path] = None

    @classmethod
    def find_or_default(cls, path: Optional[PathOrStr]) -> "TangoGlobalSettings":
        """
        Initialize the config from files by checking the default locations
        in order, or just return the default if none of the files can be found.
        """
        if path is not None:
            path = Path(path)
            if not path.is_file():
                raise FileNotFoundError(path)
            return cls.from_file(path)
        else:
            for directory in (Path("."), Path.home() / ".config"):
                for extension in ("yml", "yaml"):
                    path = directory / f"tango.{extension}"
                    if path.is_file():
                        return cls.from_file(path)
            return cls()

    @property
    def path(self) -> Optional[Path]:
        """
        The path to the file the config was read from.
        """
        return self._path

    @classmethod
    def from_file(cls, path: Path) -> "TangoGlobalSettings":
        params = Params.from_file(path)
        params["_path"] = path
        return cls.from_params(params)


@click.group(
    cls=HelpColorsGroup,
    help_options_color="green",
    help_headers_color="yellow",
    context_settings={"max_content_width": 115},
)
@click.version_option(version=VERSION)
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    help="Path to a global tango.yml settings file.",
)
@click.option(
    "--log-level",
    help="Set the global log level.",
    type=click.Choice(["debug", "info", "warning", "error"], case_sensitive=False),
    show_choices=True,
)
@click.option(
    "--no-logging",
    is_flag=True,
    help="Disable logging altogether.",
)
@click.option(
    "--file-friendly-logging",
    is_flag=True,
    help="Outputs progress bar status on separate lines and slows refresh rate.",
)
@click.pass_context
def main(
    ctx,
    config: Optional[str],
    log_level: Optional[str],
    no_logging: bool,
    file_friendly_logging: bool = False,
):
    config: TangoGlobalSettings = TangoGlobalSettings.find_or_default(config)

    if no_logging or config.no_logging:
        config.no_logging = True
        log_level = None
        config.log_level = None
    elif log_level is not None:
        config.log_level = log_level
    else:
        log_level = config.log_level

    common_logging.initialize_logging(
        log_level=log_level, file_friendly_logging=file_friendly_logging
    )

    ctx.obj = config


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
@click.pass_obj
def run(
    config: TangoGlobalSettings,
    experiment: str,
    directory: Optional[Union[str, os.PathLike]] = None,
    overrides: Optional[str] = None,
    include_package: Optional[Sequence[str]] = None,
):
    """
    Run a tango experiment.

    EXPERIMENT is the path to experiment's JSON/Jsonnet/YAML configuration file.
    """
    _run(
        config,
        experiment,
        directory=directory,
        overrides=overrides,
        include_package=include_package,
    )


@main.command(
    cls=HelpColorsCommand,
    help_options_color="green",
    help_headers_color="yellow",
    context_settings={"max_content_width": 115},
)
@click.pass_obj
def info(config: TangoGlobalSettings):
    """
    Get info about the current tango installation.
    """
    import platform

    from tango.common.util import find_integrations, import_module_and_submodules

    click.echo(f"Tango version {VERSION} (python {platform.python_version()})")

    # Show info about config.
    if config.path is not None:
        click.echo("\nConfig:")
        click.secho(f" \N{check mark} Loaded from {str(config.path)}", fg="green")
        if config.include_package:
            click.echo("\n   Included packages:")
            for package in config.include_package:
                is_found = True
                try:
                    import_module_and_submodules(package)
                except (ModuleNotFoundError, ImportError):
                    is_found = False
                if is_found:
                    click.secho(f"   \N{check mark} {package}", fg="green")
                else:
                    click.secho(f"   \N{ballot x} {package} (not found)", fg="red")

    # Show info about integrations.
    click.echo("\nIntegrations:")
    for integration in find_integrations():
        name = integration.split(".")[-1]
        is_installed = True
        try:
            import_module_and_submodules(integration)
        except (ModuleNotFoundError, ImportError):
            is_installed = False
        if is_installed:
            click.secho(f" \N{check mark} {name}", fg="green")
        else:
            click.secho(f" \N{ballot x} {name} (not installed)", fg="yellow")


def _run(
    config: TangoGlobalSettings,
    experiment: str,
    directory: Optional[Union[str, os.PathLike]] = None,
    overrides: Optional[str] = None,
    include_package: Optional[Sequence[str]] = None,
):
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
    include_package += config.include_package or []
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
