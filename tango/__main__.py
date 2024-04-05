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

``tango settings``
------------------

The ``settings`` group of commands can be used to initialize a :class:`~tango.settings.TangoGlobalSettings`
file or update fields in it.

"""
import os
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Union

import click
from click_help_colors import HelpColorsCommand, HelpColorsGroup

from tango.cli import (
    cleanup_cli,
    execute_step_graph,
    initialize_cli,
    load_settings,
    prepare_executor,
    prepare_workspace,
)
from tango.common.exceptions import CliRunError, IntegrationMissingError
from tango.common.logging import cli_logger, initialize_logging
from tango.common.params import Params
from tango.common.util import (
    find_integrations,
    import_extra_module,
    import_module_and_submodules,
)
from tango.settings import TangoGlobalSettings
from tango.step_graph import StepGraph
from tango.version import VERSION
from tango.workspace import Workspace

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


class SettingsObject(NamedTuple):
    settings: TangoGlobalSettings
    called_by_executor: bool


@click.group(name=None, **_CLICK_GROUP_DEFAULTS)
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
@click.option(
    "--called-by-executor",
    is_flag=True,
    hidden=True,
)
@click.pass_context
def main(
    ctx,
    settings: Optional[str] = None,
    log_level: Optional[str] = None,
    file_friendly_logging: bool = False,
    start_method: Optional[str] = None,
    called_by_executor: bool = False,
):
    settings: TangoGlobalSettings = load_settings(settings)

    if start_method is not None:
        settings.multiprocessing_start_method = start_method

    if log_level is not None:
        settings.log_level = log_level

    if file_friendly_logging:
        settings.file_friendly_logging = file_friendly_logging

    ctx.obj = SettingsObject(settings, called_by_executor)

    initialize_cli(settings=settings, called_by_executor=called_by_executor)


@main.result_callback()
def cleanup(*args, **kwargs):
    cleanup_cli()


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
    "-j",
    "--parallelism",
    type=int,
    help="""The maximum number of steps to run in parallel (for executors that support this).
    The exact behavior depends on the executor. If you're using the default executors,
    a value of 0 (or left unspecified) means each step is run in the main process using the default executor,
    otherwise the multicore executor is used.""",
)
@click.option(
    "-s",
    "--step-name",
    help="Execute a particular step (and its dependencies) in the experiment.",
    multiple=True,
)
@click.option(
    "-n",
    "--name",
    type=str,
    help="""Specify the name for this run.""",
)
@click.option(
    "-D",
    "--ext-var",
    type=str,
    help="""JSONNET external variables to use when loading the experiment config.
    For example, --ext-var 'pretrained_model=gpt2'.""",
    multiple=True,
)
@click.pass_obj
def run(
    obj: SettingsObject,
    experiment: str,
    workspace: Optional[str] = None,
    workspace_dir: Optional[Union[str, os.PathLike]] = None,
    overrides: Optional[str] = None,
    include_package: Optional[Sequence[str]] = None,
    parallelism: Optional[int] = None,
    step_name: Optional[Sequence[str]] = None,
    name: Optional[str] = None,
    ext_var: Optional[Sequence[str]] = None,
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
        obj.settings,
        experiment,
        workspace_url=workspace,
        overrides=overrides,
        include_package=include_package,
        parallelism=parallelism,
        step_names=step_name,
        name=name,
        called_by_executor=obj.called_by_executor,
        ext_var=ext_var,
    )


@main.command(hidden=True)
@click.argument(
    "experiment",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
)
@click.argument(
    "step_name",
    type=str,
)
@click.argument(
    "workspace_url",
    type=str,
)
@click.option(
    "-i",
    "--include-package",
    type=str,
    help="Python packages or modules to import for tango components.",
    multiple=True,
)
@click.option(
    "--log-level",
    help="Set the global log level.",
    type=click.Choice(["debug", "info", "warning", "error"], case_sensitive=False),
    show_choices=True,
)
def beaker_executor_run(
    experiment: str,
    step_name: str,
    workspace_url: str,
    include_package: Optional[Sequence[str]] = None,
    log_level: str = "debug",
):
    """
    This command is only used internally by the BeakerExecutor.
    """
    from tango.executor import Executor

    if include_package:
        for package_name in include_package:
            import_extra_module(package_name)

    # Load step graph and step.
    step_graph = StepGraph.from_file(experiment)
    step = step_graph[step_name]

    # Initialize workspace and executor.
    # NOTE: We use the default executor here because we're just running the step
    # locally in the main process.
    workspace = Workspace.from_url(workspace_url)
    executor = Executor(workspace=workspace, include_package=include_package)

    # Initialize logging.
    initialize_logging(log_level=log_level, enable_cli_logs=True, file_friendly_logging=True)

    # Run step.
    executor.execute_step(step)


@main.command(**_CLICK_COMMAND_DEFAULTS)
@click.pass_obj
def info(obj: SettingsObject):
    """
    Get info about the current tango installation.
    """
    import platform

    cli_logger.info("Tango version %s (python %s)", VERSION, platform.python_version())
    cli_logger.info("")

    # Show info about settings.
    if obj.settings.path is not None:
        cli_logger.info("[underline]Settings:[/]")
        cli_logger.info("[green] \N{check mark} Loaded from %s[/]", obj.settings.path)
        if obj.settings.include_package:
            cli_logger.info("   Included packages:")
            for package in obj.settings.include_package:
                is_found = True
                try:
                    import_module_and_submodules(package)
                except (ModuleNotFoundError, ImportError):
                    is_found = False
                if is_found:
                    cli_logger.info("   [green]\N{check mark} %s[/]", package)
                else:
                    cli_logger.info("   [red]\N{ballot x} %s (not found)[/]", package)
        cli_logger.info("")

    # Show info about integrations.
    cli_logger.info("[underline]Integrations:[/]")
    for integration in find_integrations():
        name = integration.split(".")[-1]
        is_installed = True
        try:
            import_module_and_submodules(integration, recursive=False)
        except (IntegrationMissingError, ModuleNotFoundError, ImportError):
            is_installed = False
        if is_installed:
            cli_logger.info(" [green]\N{check mark} %s[/]", name)
        else:
            cli_logger.info(" [yellow]\N{ballot x} %s (not installed)[/]", name)


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
def init(obj: SettingsObject, path: Optional[str] = None, force: bool = False):
    """
    Initialize the settings file.
    """
    path_to_write = Path(path or TangoGlobalSettings._DEFAULT_LOCATION)
    if path_to_write.is_file() and not force:
        raise click.ClickException("Settings file already exists! Use -f/--force to overwrite it.")
    obj.settings.to_file(path_to_write)
    cli_logger.info(
        "[green]\N{check mark} Settings file written to [bold]%s[/bold][/green]", path_to_write
    )


@settings.group(name="set", **_CLICK_GROUP_DEFAULTS)
@click.pass_obj
def set_setting(obj: SettingsObject):
    """
    Set a value in the settings file.
    """
    if obj.settings.path is None:
        raise click.ClickException(
            "Settings file not found! Did you forget to call 'tango settings init'?"
        )


@set_setting.result_callback()
def save_settings(settings: TangoGlobalSettings):
    settings.save()


@set_setting.command(**_CLICK_COMMAND_DEFAULTS)
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
def workspace(obj: SettingsObject, workspace: str, validate: bool = True) -> TangoGlobalSettings:
    """
    Set the default workspace path or URL.
    """
    from urllib.parse import urlparse

    if not urlparse(workspace).scheme:
        obj.settings.workspace = {"type": "local", "dir": str(Path(workspace).resolve())}
    else:
        obj.settings.workspace = {"type": "from_url", "url": workspace}

    if validate:
        for package_name in obj.settings.include_package or []:
            import_extra_module(package_name)

        Workspace.from_params(obj.settings.workspace.copy())

    return obj.settings


@set_setting.command(**_CLICK_COMMAND_DEFAULTS)
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
    obj: SettingsObject,
    packages: List[str],
    append: bool = False,
    validate: bool = True,
) -> TangoGlobalSettings:
    """
    Set or add modules to automatically import on 'tango run'.
    """
    new_include: List[str]
    if append:
        new_include = obj.settings.include_package or []
    else:
        new_include = []
    for package in packages:
        if package not in new_include:
            new_include.append(package)
    obj.settings.include_package = new_include
    if validate:
        for package in obj.settings.include_package:
            try:
                import_module_and_submodules(package)
            except (ModuleNotFoundError, ImportError):
                raise click.ClickException(f"Failed to import '{package}'")
    return obj.settings


@set_setting.command(**_CLICK_COMMAND_DEFAULTS)
@click.argument(
    "level",
    type=click.Choice(["debug", "info", "warning", "error"], case_sensitive=False),
)
@click.pass_obj
def log_level(obj: SettingsObject, level: str) -> TangoGlobalSettings:
    """
    Set the log level.
    """
    obj.settings.log_level = level.lower()
    return obj.settings


@set_setting.command(**_CLICK_COMMAND_DEFAULTS)
@click.argument(
    "value",
    type=bool,
)
@click.pass_obj
def file_friendly_logging(obj: SettingsObject, value: bool) -> TangoGlobalSettings:
    """
    Toggle file friendly logging mode.
    """
    obj.settings.file_friendly_logging = value
    return obj.settings


@set_setting.command(**_CLICK_COMMAND_DEFAULTS)
@click.argument(
    "start_method",
    type=click.Choice(["fork", "spawn", "forkserver"], case_sensitive=True),
)
@click.pass_obj
def multiprocessing_start_method(obj: SettingsObject, start_method: str) -> TangoGlobalSettings:
    """
    Set the Python multiprocessing start method.
    """
    obj.settings.multiprocessing_start_method = start_method
    return obj.settings


@set_setting.command(**_CLICK_COMMAND_DEFAULTS)
@click.argument(
    "key",
    type=str,
)
@click.argument(
    "value",
    type=str,
)
@click.pass_obj
def env(obj: SettingsObject, key: str, value: str) -> TangoGlobalSettings:
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

    if obj.settings.environment is None:
        obj.settings.environment = {}
    obj.settings.environment[key] = value
    return obj.settings


def _run(
    settings: TangoGlobalSettings,
    experiment: str,
    workspace_url: Optional[str] = None,
    overrides: Optional[str] = None,
    include_package: Optional[Sequence[str]] = None,
    step_names: Optional[Sequence[str]] = None,
    parallelism: Optional[int] = None,
    multicore: Optional[bool] = None,
    name: Optional[str] = None,
    called_by_executor: bool = False,
    ext_var: Optional[Sequence[str]] = None,
) -> str:
    # Read params.
    ext_vars: Dict[str, str] = {}
    for var in ext_var or []:
        try:
            key, value = var.split("=")
        except ValueError:
            raise CliRunError(f"Invalid --ext-var '{var}'")
        ext_vars[key] = value
    params = Params.from_file(experiment, params_overrides=overrides or "", ext_vars=ext_vars)

    # Import included packages to find registered components.
    # NOTE: The Executor imports these as well because it's meant to be used
    # directly, but we also need to import here in case the user is using a
    # custom Executor, StepCache, or Workspace.
    include_package: List[str] = list(include_package or [])
    include_package += params.pop("include_package", [])
    include_package += settings.include_package or []
    for package_name in include_package:
        import_extra_module(package_name)

    # Initialize step graph.
    step_graph: StepGraph = StepGraph.from_params(params.pop("steps"))
    params.assert_empty("'tango run'")

    if step_names:
        for step_name in step_names:
            assert step_name in step_graph, (
                f"You want to run a step called '{step_name}', but it cannot be found in the experiment config. "
                f"The config contains: {list(step_graph.keys())}."
            )
        step_graph = step_graph.sub_graph(*step_names)

    # Execute step graph in workspace

    workspace = prepare_workspace(settings=settings, workspace_url=workspace_url)

    executor = prepare_executor(
        settings=settings,
        workspace=workspace,
        include_package=include_package,
        parallelism=parallelism,
        multicore=multicore,
        called_by_executor=called_by_executor,
    )

    run_name = execute_step_graph(
        step_graph=step_graph,
        workspace=workspace,
        executor=executor,
        name=name,
        called_by_executor=called_by_executor,
        step_names=step_names,
    )

    return run_name


if __name__ == "__main__":
    main()
