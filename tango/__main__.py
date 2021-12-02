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
      --file-friendly-logging         Outputs progress bar status on separate lines and slows refresh rate.
      --help                          Show this message and exit.

    Commands:
      info    Get info about the current tango installation
      run     Run a tango experiment
      server  Run a local webserver that watches a workspace

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

``tango server``
----------------

The ``server`` command spins up a web server that watches a workspace. You can use this to track the
progress of your runs while they are happening.

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
from tango.common.util import import_extra_module
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

    log_level: Optional[str] = "error"
    """
    The log level to use. Options are "debug", "info", "warning", and "error".
    """

    file_friendly_logging: bool = False
    """
    If this flag is set to ``True``, we add newlines to tqdm output, even on an interactive terminal, and we slow
    down tqdm's output to only once every 10 seconds.

    By default, it is set to ``False``.
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
    "--file-friendly-logging",
    is_flag=True,
    help="Outputs progress bar status on separate lines and slows refresh rate.",
)
@click.pass_context
def main(
    ctx,
    config: Optional[str],
    log_level: Optional[str],
    file_friendly_logging: bool = False,
):
    config: TangoGlobalSettings = TangoGlobalSettings.find_or_default(config)

    if log_level is not None:
        config.log_level = log_level

    if file_friendly_logging is not None:
        config.file_friendly_logging = file_friendly_logging

    common_logging.initialize_logging(
        log_level=config.log_level,
        file_friendly_logging=config.file_friendly_logging,
        enable_click_logs=True,
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
    "--workspace-dir",
    type=click.Path(file_okay=False),
    help="""The directory of the workspace in which to work. If not specified,
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
    "--server/--no-server",
    type=bool,
    help="Start a server that visualizes the current run",
    default=True,
)
@click.pass_obj
def run(
    config: TangoGlobalSettings,
    experiment: str,
    workspace_dir: Optional[Union[str, os.PathLike]] = None,
    overrides: Optional[str] = None,
    include_package: Optional[Sequence[str]] = None,
    server: bool = True,
):
    """
    Run a tango experiment

    EXPERIMENT is the path to experiment's JSON/Jsonnet/YAML configuration file.
    """
    _run(
        config,
        experiment,
        workspace_dir=workspace_dir,
        overrides=overrides,
        include_package=include_package,
        start_server=server,
    )


@main.command(
    cls=HelpColorsCommand,
    help_options_color="green",
    help_headers_color="yellow",
    context_settings={"max_content_width": 115},
)
@click.option(
    "-d",
    "--workspace-dir",
    type=click.Path(file_okay=False),
    help="""The directory of the workspace to monitor.""",
)
def server(workspace_dir: Union[str, os.PathLike]):
    """
    Run a local webserver that watches a workspace
    """
    from tango.local_workspace import LocalWorkspace
    from tango.server.workspace_server import WorkspaceServer

    workspace_dir = Path(workspace_dir)
    workspace = LocalWorkspace(workspace_dir)
    server = WorkspaceServer.on_free_port(workspace)
    common_logging.click_logger.info(
        "Server started at " + click.style(server.address_for_display(), bold=True)
    )
    server.serve_forever()


@main.command(
    cls=HelpColorsCommand,
    help_options_color="green",
    help_headers_color="yellow",
    context_settings={"max_content_width": 115},
)
@click.pass_obj
def info(config: TangoGlobalSettings):
    """
    Get info about the current tango installation
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
    workspace_dir: Optional[Union[str, os.PathLike]] = None,
    overrides: Optional[str] = None,
    include_package: Optional[Sequence[str]] = None,
    start_server: bool = True,
) -> Path:
    from tango.executor import Executor
    from tango.local_workspace import LocalWorkspace
    from tango.server.workspace_server import WorkspaceServer
    from tango.step_graph import StepGraph

    # Read params.
    params = Params.from_file(experiment, params_overrides=overrides or "")

    # Import included packages to find registered components.
    # NOTE: The Executor imports these as well because it's meant to be used
    # directly, but we also need to import here in case the user is using a
    # custom Executor, StepCache, or Workspace.
    include_package: List[str] = list(include_package or [])
    include_package += params.pop("include_package", [])
    include_package += config.include_package or []
    for package_name in include_package:
        import_extra_module(package_name)

    # Prepare directory.
    if workspace_dir is None:
        from tempfile import mkdtemp

        workspace_dir = mkdtemp(prefix="tango-")
        click.echo(
            "Creating temporary directory for run: " + click.style(f"{workspace_dir}", fg="yellow")
        )
    workspace_dir = Path(workspace_dir)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    workspace = LocalWorkspace(workspace_dir)

    # Initialize step graph, server, and executor.
    step_graph = StepGraph(params.pop("steps", keep_as_dict=True))
    server = None
    if start_server:
        server = WorkspaceServer.on_free_port(workspace)
        server.serve_in_background()

    executor = Executor(workspace=workspace, include_package=include_package, server=server)

    # Now execute the step graph.
    run_name = executor.execute_step_graph(step_graph)
    return workspace.run_dir(run_name)


if __name__ == "__main__":
    main()
