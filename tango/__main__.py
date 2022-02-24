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
      --settings FILE                 Path to a global tango.yml settings file.
      --log-level [debug|info|warning|error]
                                      Set the global log level.
      --file-friendly-logging         Outputs progress bar status on separate lines and slows refresh rate.
      --start-method [fork|spawn|forkserver]
                                      Set the multiprocessing start method.
      --help                          Show this message and exit.

    Commands:
      info      Get info about the current tango installation.
      run       Run a tango experiment.
      server    Run a local webserver that watches a workspace.
      settings  Commands for initializing and updating global settings.

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

``tango settings``
------------------

The ``settings`` group of commands can be used to initialize a :class:`~tango.settings.TangoGlobalSettings`
file or update fields in it.

"""
import multiprocessing as mp
import os
import warnings
from pathlib import Path
from typing import List, Optional, Sequence, Union

import click
from click_help_colors import HelpColorsCommand, HelpColorsGroup

from tango.common.logging import click_logger, initialize_logging, teardown_logging
from tango.common.params import Params
from tango.common.util import (
    find_integrations,
    import_extra_module,
    import_module_and_submodules,
)
from tango.settings import TangoGlobalSettings
from tango.version import VERSION

_CLICK_GROUP_DEFAULTS = {
    "cls": HelpColorsGroup,
    "help_options_color": "green",
    "help_headers_color": "yellow",
    "context_settings": {"max_content_width": 115},
}

_CLICK_COMMAND_DEFAULTS = {
    "cls": HelpColorsCommand,
    "help_options_color": "green",
    "help_headers_color": "yellow",
    "context_settings": {"max_content_width": 115},
}


@click.group(**_CLICK_GROUP_DEFAULTS)
@click.version_option(version=VERSION)
@click.option(
    "--settings",
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
    settings: Optional[str] = None,
    log_level: Optional[str] = None,
    file_friendly_logging: bool = False,
    start_method: Optional[str] = None,
):
    settings: TangoGlobalSettings = (
        TangoGlobalSettings.from_file(settings)
        if settings is not None
        else TangoGlobalSettings.default()
    )

    if settings.environment:
        from tango.common.aliases import EnvVarNames

        # These environment variables should not be set this way since they'll be ignored.
        blocked_env_variable_names = EnvVarNames.values()

        for key, value in settings.environment.items():
            if key not in blocked_env_variable_names:
                os.environ[key] = value
            else:
                warnings.warn(
                    f"Ignoring environment variable '{key}' from settings file. "
                    f"Please use the corresponding settings field instead.",
                    UserWarning,
                )

    if start_method is not None:
        settings.multiprocessing_start_method = start_method

    mp.set_start_method(settings.multiprocessing_start_method)

    if log_level is not None:
        settings.log_level = log_level

    if file_friendly_logging:
        settings.file_friendly_logging = file_friendly_logging

    initialize_logging(
        log_level=settings.log_level,
        file_friendly_logging=settings.file_friendly_logging,
        enable_click_logs=True,
    )

    ctx.obj = settings


@main.result_callback()
def cleanup(*args, **kwargs):
    teardown_logging()


@main.command(**_CLICK_COMMAND_DEFAULTS)
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
@click.pass_obj
def run(
    settings: TangoGlobalSettings,
    experiment: str,
    workspace: Optional[str] = None,
    workspace_dir: Optional[Union[str, os.PathLike]] = None,
    overrides: Optional[str] = None,
    include_package: Optional[Sequence[str]] = None,
    server: bool = True,
):
    """
    Run a tango experiment.

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
        settings,
        experiment,
        workspace_url=workspace,
        overrides=overrides,
        include_package=include_package,
        start_server=server,
    )


@main.command(**_CLICK_COMMAND_DEFAULTS)
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
    settings: TangoGlobalSettings,
    workspace: Optional[str],
    workspace_dir: Optional[Union[str, os.PathLike]] = None,
):
    """
    Run a local webserver that watches a workspace.
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
    elif settings.workspace is not None:
        workspace_to_watch = Workspace.from_params(settings.workspace)
    else:
        raise click.ClickException(
            "-w/--workspace or -d/--workspace-dir required unless a default workspace is specified "
            "in tango settings file."
        )

    server = WorkspaceServer.on_free_port(workspace_to_watch)
    click_logger.info("Server started at " + click.style(server.address_for_display(), bold=True))
    server.serve_forever()


@main.command(**_CLICK_COMMAND_DEFAULTS)
@click.pass_obj
def info(settings: TangoGlobalSettings):
    """
    Get info about the current tango installation.
    """
    import platform

    click_logger.info(f"Tango version {VERSION} (python {platform.python_version()})")

    # Show info about settings.
    if settings.path is not None:
        click_logger.info("\nSettings:")
        click_logger.info(
            click.style(f" \N{check mark} Loaded from {str(settings.path)}", fg="green")
        )
        if settings.include_package:
            click_logger.info("\n   Included packages:")
            for package in settings.include_package:
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


@main.group(**_CLICK_GROUP_DEFAULTS)
@click.pass_obj
def settings(ctx):
    """
    Commands for initializing and updating global settings.
    """


@settings.command(**_CLICK_COMMAND_DEFAULTS)
@click.option(
    "-p",
    "--path",
    type=click.Path(exists=False, dir_okay=False, resolve_path=True),
    default=None,
    help="""The path to write the settings to.""",
)
@click.option(
    "-f",
    "--force",
    is_flag=True,
    help="""Force overwrite the file if it exists.""",
)
@click.pass_obj
def init(settings: TangoGlobalSettings, path: Optional[str] = None, force: bool = False):
    """
    Initialize the settings file.
    """
    path_to_write = Path(path or TangoGlobalSettings._DEFAULT_LOCATION)
    if path_to_write.is_file() and not force:
        raise click.ClickException("Settings file already exists! Use -f/--force to overwrite it.")
    settings.to_file(path_to_write)
    click_logger.info(
        click.style("\N{check mark} Settings file written to ", fg="green")
        + click.style(path_to_write, fg="green", bold=True)
    )


@settings.group(**_CLICK_GROUP_DEFAULTS)
@click.pass_obj
def set(settings: TangoGlobalSettings):
    """
    Set a value in the settings file.
    """
    if settings.path is None:
        raise click.ClickException(
            "Settings file not found! Did you forget to call 'tango settings init'?"
        )


@set.result_callback()
def save_settings(settings: TangoGlobalSettings):
    settings.save()


@set.command(**_CLICK_COMMAND_DEFAULTS)
@click.argument(
    "workspace",
    type=str,
)
@click.option(
    "--validate/--no-validate",
    type=bool,
    help="Validate that the workspace can be initialized.",
    default=True,
)
@click.pass_obj
def workspace(
    settings: TangoGlobalSettings, workspace: str, validate: bool = True
) -> TangoGlobalSettings:
    """
    Set the default workspace path or URL.
    """
    from urllib.parse import urlparse

    if not urlparse(workspace).scheme:
        settings.workspace = {"type": "local", "dir": str(Path(workspace).resolve())}
    else:
        settings.workspace = {"type": "from_url", "url": workspace}

    if validate:
        from tango.workspace import Workspace

        for package_name in settings.include_package or []:
            import_extra_module(package_name)

        Workspace.from_params(settings.workspace.copy())

    return settings


@set.command(**_CLICK_COMMAND_DEFAULTS)
@click.argument(
    "packages",
    type=str,
    nargs=-1,
)
@click.option(
    "-a",
    "--append",
    is_flag=True,
    help="Appends packages instead of overwriting.",
)
@click.option(
    "--validate/--no-validate",
    type=bool,
    help="Validate that the workspace can be initialized.",
    default=True,
)
@click.pass_obj
def include_package(
    settings: TangoGlobalSettings,
    packages: List[str],
    append: bool = False,
    validate: bool = True,
) -> TangoGlobalSettings:
    """
    Set or add modules to automatically import on 'tango run'.
    """
    new_include: List[str]
    if append:
        new_include = settings.include_package or []
    else:
        new_include = []
    for package in packages:
        if package not in new_include:
            new_include.append(package)
    settings.include_package = new_include
    if validate:
        for package in settings.include_package:
            try:
                import_module_and_submodules(package)
            except (ModuleNotFoundError, ImportError):
                raise click.ClickException(f"Failed to import '{package}'")
    return settings


@set.command(**_CLICK_COMMAND_DEFAULTS)
@click.argument(
    "level",
    type=click.Choice(["debug", "info", "warning", "error"], case_sensitive=False),
)
@click.pass_obj
def log_level(settings: TangoGlobalSettings, level: str) -> TangoGlobalSettings:
    """
    Set the log level.
    """
    settings.log_level = level.lower()
    return settings


@set.command(**_CLICK_COMMAND_DEFAULTS)
@click.argument(
    "value",
    type=bool,
)
@click.pass_obj
def file_friendly_logging(settings: TangoGlobalSettings, value: bool) -> TangoGlobalSettings:
    """
    Toggle file friendly logging mode.
    """
    settings.file_friendly_logging = value
    return settings


@set.command(**_CLICK_COMMAND_DEFAULTS)
@click.argument(
    "start_method",
    type=click.Choice(["fork", "spawn", "forkserver"], case_sensitive=True),
)
@click.pass_obj
def multiprocessing_start_method(
    settings: TangoGlobalSettings, start_method: str
) -> TangoGlobalSettings:
    """
    Set the Python multiprocessing start method.
    """
    settings.multiprocessing_start_method = start_method
    return settings


@set.command(**_CLICK_COMMAND_DEFAULTS)
@click.argument(
    "key",
    type=str,
)
@click.argument(
    "value",
    type=str,
)
@click.pass_obj
def env(settings: TangoGlobalSettings, key: str, value: str) -> TangoGlobalSettings:
    """
    Add or update an environment variable.
    """
    from tango.common.aliases import EnvVarNames

    # These environment variables should not be set this way since they'll be ignored.
    blocked_env_variable_names = EnvVarNames.values()

    if key in blocked_env_variable_names:
        raise click.ClickException(
            f"Cannot add environment variable '{key}' to settings. "
            f"Please set the corresponding settings field instead."
        )

    if settings.environment is None:
        settings.environment = {}
    settings.environment[key] = value
    return settings


def _run(
    settings: TangoGlobalSettings,
    experiment: str,
    workspace_url: Optional[str] = None,
    overrides: Optional[str] = None,
    include_package: Optional[Sequence[str]] = None,
    start_server: bool = True,
) -> str:
    from tango.executor import Executor
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
    include_package += settings.include_package or []
    for package_name in include_package:
        import_extra_module(package_name)

    # Prepare workspace.
    workspace: Workspace
    if workspace_url is not None:
        workspace = Workspace.from_url(workspace_url)
    elif settings.workspace is not None:
        workspace = Workspace.from_params(settings.workspace)
    else:
        workspace = default_workspace

    # Initialize step graph and register run.
    step_graph = StepGraph(params.pop("steps", keep_as_dict=True))
    params.assert_empty("'tango run'")
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
        executor = Executor(workspace=workspace, include_package=include_package)
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
            click.style("Finished run ", fg="green") + click.style(run.name, fg="green", bold=True)
        )

    return run.name


if __name__ == "__main__":
    main()
