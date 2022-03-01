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
      --start-method [fork|spawn|forkserver]
                                      Set the multiprocessing start method.
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
import multiprocessing as mp
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import click
from click_help_colors import HelpColorsCommand, HelpColorsGroup

from tango.common.aliases import PathOrStr
from tango.common.from_params import FromParams
from tango.common.logging import click_logger, initialize_logging, teardown_logging
from tango.common.params import Params
from tango.common.util import import_extra_module
from tango.version import VERSION


@dataclass
class TangoGlobalSettings(FromParams):
    """
    Defines global settings for tango.
    """

    workspace: Optional[Dict[str, Any]] = None
    """
    Parameters to initialize a :class:`tango.workspace.Workspace` with.
    """

    include_package: Optional[List[str]] = None
    """
    An list of modules where custom registered steps or classes can be found.
    """

    log_level: Optional[str] = "warning"
    """
    The log level to use. Options are "debug", "info", "warning", and "error".

    .. note::
        This does not affect the :data:`~tango.common.logging.click_logger`
        or logs from :class:`~tango.common.Tqdm` progress bars.

    """

    file_friendly_logging: Optional[bool] = None
    """
    If this flag is set to ``True``, we add newlines to tqdm output, even on an interactive terminal, and we slow
    down tqdm's output to only once every 10 seconds.
    """

    multiprocessing_start_method: str = "spawn"
    """
    The ``start_method`` to use when starting new multiprocessing workers. Can be "fork", "spawn",
    or "forkserver". Default is "spawn".

    See :func:`multiprocessing.set_start_method()` for more details.
    """

    _path: Optional[Path] = None

    @classmethod
    def default(cls) -> "TangoGlobalSettings":
        """
        Initialize the config from files by checking the default locations
        in order, or just return the default if none of the files can be found.
        """
        for directory in (Path("."), Path.home() / ".config"):
            for extension in ("yml", "yaml"):
                path = directory / f"tango.{extension}"
                if path.is_file():
                    return cls.from_file(path)
        return cls()

    @classmethod
    def find_or_default(cls, path: Optional[PathOrStr]) -> "TangoGlobalSettings":
        """
        Initialize the config from a given configuration file, or falls back to returning
        the default configuration if no file is given.
        """
        if path is not None:
            path = Path(path)
            if not path.is_file():
                raise FileNotFoundError(path)
            return cls.from_file(path)
        else:
            return cls.default()

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
@click.option(
    "--start-method",
    help="Set the multiprocessing start method.",
    type=click.Choice(["fork", "spawn", "forkserver"], case_sensitive=True),
    show_choices=True,
)
@click.pass_context
def main(
    ctx,
    config: Optional[str] = None,
    log_level: Optional[str] = None,
    file_friendly_logging: bool = False,
    start_method: Optional[str] = None,
):
    config: TangoGlobalSettings = TangoGlobalSettings.find_or_default(config)

    if start_method is not None:
        config.multiprocessing_start_method = start_method

    mp.set_start_method(config.multiprocessing_start_method)

    if log_level is not None:
        config.log_level = log_level

    if file_friendly_logging:
        config.file_friendly_logging = file_friendly_logging

    initialize_logging(
        log_level=config.log_level,
        file_friendly_logging=config.file_friendly_logging,
        enable_click_logs=True,
    )

    ctx.obj = config


@main.result_callback()
def cleanup(*args, **kwargs):
    teardown_logging()


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
    "-w",
    "--workspace",
    type=click.Path(file_okay=False),
    help="""A workspace path or URL. If not specified, the workspace from any global tango
    settings file will be used, if found, otherwise an ephemeral MemoryWorkspace.""",
    default=None,
)
@click.option(
    "-d",
    "--workspace-dir",
    type=click.Path(file_okay=False),
    default=None,
    hidden=True,
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
@click.option(
    "-s",
    "--step-name",
    help="Execute a particular step (and its dependencies) in the experiment.",
    type=str,
    default=None,
)
@click.pass_obj
def run(
    config: TangoGlobalSettings,
    experiment: str,
    workspace: Optional[str] = None,
    workspace_dir: Optional[Union[str, os.PathLike]] = None,
    overrides: Optional[str] = None,
    include_package: Optional[Sequence[str]] = None,
    server: bool = True,
    step_name: Optional[str] = None,
):
    """
    Run a tango experiment

    EXPERIMENT is the path to experiment's JSON/Jsonnet/YAML configuration file.
    """
    if workspace_dir is not None:
        import warnings

        warnings.warn(
            "-d/--workspace-dir option is deprecated. Please use -w/--workspace instead.",
            DeprecationWarning,
        )

        if workspace is not None:
            raise click.ClickException(
                "-w/--workspace is mutually exclusive with -d/--workspace-dir"
            )

        workspace = "local://" + str(workspace_dir)

    _run(
        config,
        experiment,
        workspace_url=workspace,
        overrides=overrides,
        include_package=include_package,
        start_server=server,
        step_name=step_name,
    )


@main.command(
    cls=HelpColorsCommand,
    help_options_color="green",
    help_headers_color="yellow",
    context_settings={"max_content_width": 115},
)
@click.option(
    "-w",
    "--workspace",
    type=click.Path(file_okay=False),
    help="""A workspace URL. If not specified, the workspace from any global tango
    settings file will be used, if found, otherwise an ephemeral MemoryWorkspace.""",
    default=None,
)
@click.option(
    "-d",
    "--workspace-dir",
    type=click.Path(file_okay=False),
    default=None,
    hidden=True,
)
@click.pass_obj
def server(
    config: TangoGlobalSettings,
    workspace: Optional[str],
    workspace_dir: Optional[Union[str, os.PathLike]] = None,
):
    """
    Run a local webserver that watches a workspace
    """
    from tango.server.workspace_server import WorkspaceServer
    from tango.workspace import Workspace
    from tango.workspaces import LocalWorkspace

    workspace_to_watch: Workspace
    if workspace_dir is not None:
        if workspace is not None:
            raise click.ClickException(
                "-w/--workspace is mutually exclusive with -d/--workspace-dir"
            )
        workspace_to_watch = LocalWorkspace(workspace_dir)
    elif workspace is not None:
        workspace_to_watch = Workspace.from_url(workspace)
    elif config.workspace is not None:
        workspace_to_watch = Workspace.from_params(config.workspace)
    else:
        raise click.ClickException(
            "-w/--workspace or -d/--workspace-dir required unless a default workspace is specified "
            "in tango settings file."
        )

    server = WorkspaceServer.on_free_port(workspace_to_watch)
    click_logger.info("Server started at " + click.style(server.address_for_display(), bold=True))
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

    click_logger.info(f"Tango version {VERSION} (python {platform.python_version()})")

    # Show info about config.
    if config.path is not None:
        click_logger.info("\nConfig:")
        click_logger.info(
            click.style(f" \N{check mark} Loaded from {str(config.path)}", fg="green")
        )
        if config.include_package:
            click_logger.info("\n   Included packages:")
            for package in config.include_package:
                is_found = True
                try:
                    import_module_and_submodules(package)
                except (ModuleNotFoundError, ImportError):
                    is_found = False
                if is_found:
                    click_logger.info(click.style(f"   \N{check mark} {package}", fg="green"))
                else:
                    click_logger.info(
                        click.style(f"   \N{ballot x} {package} (not found)", fg="red")
                    )

    # Show info about integrations.
    click_logger.info("\nIntegrations:")
    for integration in find_integrations():
        name = integration.split(".")[-1]
        is_installed = True
        try:
            import_module_and_submodules(integration)
        except (ModuleNotFoundError, ImportError):
            is_installed = False
        if is_installed:
            click_logger.info(click.style(f" \N{check mark} {name}", fg="green"))
        else:
            click_logger.info(click.style(f" \N{ballot x} {name} (not installed)", fg="yellow"))


def _run(
    config: TangoGlobalSettings,
    experiment: str,
    workspace_url: Optional[str] = None,
    overrides: Optional[str] = None,
    include_package: Optional[Sequence[str]] = None,
    start_server: bool = True,
    step_name: Optional[str] = None,
) -> str:
    from tango.executor import Executor
    from tango.multicore_executor import MulticoreExecutor
    from tango.server.workspace_server import WorkspaceServer
    from tango.step_graph import StepGraph
    from tango.workspace import Workspace
    from tango.workspaces import default_workspace

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

    # Prepare workspace.
    workspace: Workspace
    if workspace_url is not None:
        workspace = Workspace.from_url(workspace_url)
    elif config.workspace is not None:
        workspace = Workspace.from_params(config.workspace)
    else:
        workspace = default_workspace

    # Initialize step graph and register run.
    step_graph = StepGraph.from_params(params.pop("steps", keep_as_dict=True))
    params.assert_empty("'tango run'")

    if step_name is not None:
        assert step_name in step_graph, (
            f"You want to run a step called '{step_name}', but it cannot be found in the experiment config. "
            f"The config contains: {list(step_graph.keys())}."
        )
        # step_graph = step_graph.get_sub_graph(step_name)

    run = workspace.register_run(step for step in step_graph.values())

    # Capture logs to file.
    with workspace.capture_logs_for_run(run.name):
        click_logger.info(
            click.style("Starting new run ", fg="green")
            + click.style(run.name, fg="green", bold=True)
        )

        # Initialize server.
        if start_server:
            server = WorkspaceServer.on_free_port(workspace)
            server.serve_in_background()
            click_logger.info(
                "Server started at " + click.style(server.address_for_display(run.name), bold=True)
            )

        # Initialize Executor and execute the step graph.
        # executor = Executor(workspace=workspace, include_package=include_package)
        executor = MulticoreExecutor(
            workspace=workspace, include_package=include_package, parallelism=2
        )
        if step_name is not None:
            uncacheable_leaf_steps = step_graph.find_uncacheable_leaf_steps()
            executor.execute_step(step_graph[step_name], step_name in uncacheable_leaf_steps)
        else:
            executor.execute_step_graph(step_graph)

            # Print everything that has been computed.
            ordered_steps = sorted(step_graph.values(), key=lambda step: step.name)
            for step in ordered_steps:
                info = workspace.step_info(step)
                if info.result_location is not None:
                    click_logger.info(
                        click.style("\N{check mark} The output for ", fg="green")
                        + click.style(f'"{step.name}"', bold=True, fg="green")
                        + click.style(" is in ", fg="green")
                        + click.style(f"{info.result_location}", bold=True, fg="green")
                    )

            click_logger.info(
                click.style("Finished run ", fg="green")
                + click.style(run.name, fg="green", bold=True)
            )

    return run.name


if __name__ == "__main__":
    main()
