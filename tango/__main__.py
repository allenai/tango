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
import logging
import multiprocessing as mp
import os
import sys
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Union

import click
from click_help_colors import HelpColorsCommand, HelpColorsGroup

from tango.common.exceptions import CliRunError, IntegrationMissingError
from tango.common.logging import (
    cli_logger,
    initialize_logging,
    initialize_prefix_logging,
    teardown_logging,
)
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

if TYPE_CHECKING:
    from tango.executor import ExecutorOutput
    from tango.workspace import Run

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

_CALLED_BY_EXECUTOR: bool = (
    False  # Flag used internally to determine if CLI was called by the MulticoreExecutor.
)


logger = logging.getLogger(__name__)


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
    if not sys.warnoptions:
        warnings.simplefilter("default", category=DeprecationWarning)

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

    if called_by_executor:
        # We only set this flag instead of calling `initialize_prefix_logging` here
        # because we do not know the `step_name` yet.
        global _CALLED_BY_EXECUTOR
        _CALLED_BY_EXECUTOR = True
    else:
        initialize_logging(
            log_level=settings.log_level,
            file_friendly_logging=settings.file_friendly_logging,
            enable_cli_logs=True,
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
    settings: TangoGlobalSettings,
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
        settings,
        experiment,
        workspace_url=workspace,
        overrides=overrides,
        include_package=include_package,
        parallelism=parallelism,
        step_names=step_name,
        name=name,
        called_by_executor=_CALLED_BY_EXECUTOR,
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
def info(settings: TangoGlobalSettings):
    """
    Get info about the current tango installation.
    """
    import platform

    cli_logger.info("Tango version %s (python %s)", VERSION, platform.python_version())
    cli_logger.info("")

    # Show info about settings.
    if settings.path is not None:
        cli_logger.info("[underline]Settings:[/]")
        cli_logger.info("[green] \N{check mark} Loaded from %s[/]", settings.path)
        if settings.include_package:
            cli_logger.info("   Included packages:")
            for package in settings.include_package:
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
def init(settings: TangoGlobalSettings, path: Optional[str] = None, force: bool = False):
    """
    Initialize the settings file.
    """
    path_to_write = Path(path or TangoGlobalSettings._DEFAULT_LOCATION)
    if path_to_write.is_file() and not force:
        raise click.ClickException("Settings file already exists! Use -f/--force to overwrite it.")
    settings.to_file(path_to_write)
    cli_logger.info(
        "[green]\N{check mark} Settings file written to [bold]%s[/bold][/green]", path_to_write
    )


@settings.group(name="set", **_CLICK_GROUP_DEFAULTS)
@click.pass_obj
def set_setting(settings: TangoGlobalSettings):
    """
    Set a value in the settings file.
    """
    if settings.path is None:
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
        for package_name in settings.include_package or []:
            import_extra_module(package_name)

        Workspace.from_params(settings.workspace.copy())

    return settings


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


@set_setting.command(**_CLICK_COMMAND_DEFAULTS)
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


@set_setting.command(**_CLICK_COMMAND_DEFAULTS)
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


@set_setting.command(**_CLICK_COMMAND_DEFAULTS)
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
    step_names: Optional[Sequence[str]] = None,
    parallelism: Optional[int] = None,
    multicore: Optional[bool] = None,
    name: Optional[str] = None,
    called_by_executor: bool = False,
    ext_var: Optional[Sequence[str]] = None,
) -> str:
    from tango.executor import Executor
    from tango.executors import MulticoreExecutor
    from tango.workspaces import MemoryWorkspace, default_workspace

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

    # Prepare workspace.
    workspace: Workspace
    if workspace_url is not None:
        workspace = Workspace.from_url(workspace_url)
    elif settings.workspace is not None:
        workspace = Workspace.from_params(settings.workspace)
    else:
        workspace = default_workspace

    executor: Executor
    if not called_by_executor and settings.executor is not None:
        if multicore is not None:
            logger.warning(
                "Ignoring argument 'multicore' since executor is defined in %s",
                settings.path or "setting",
            )
        executor = Executor.from_params(
            settings.executor,
            workspace=workspace,
            include_package=include_package,
            **(dict(parallelism=parallelism) if parallelism is not None else {}),  # type: ignore
        )
    else:
        # Determine if we can use the multicore executor.
        if multicore is None:
            if isinstance(workspace, MemoryWorkspace):
                # Memory workspace does not work with multiple cores.
                multicore = False
            elif "pydevd" in sys.modules:
                # Pydevd doesn't reliably follow child processes, so we disable multicore under the debugger.
                logger.warning("Debugger detected, disabling multicore.")
                multicore = False
            elif parallelism is None or parallelism == 0:
                multicore = False
            else:
                multicore = True

        if multicore:
            executor = MulticoreExecutor(
                workspace=workspace, include_package=include_package, parallelism=parallelism
            )
        else:
            executor = Executor(workspace=workspace, include_package=include_package)

    # Initialize step graph.
    step_graph = StepGraph.from_params(params.pop("steps"))
    sub_graph: Optional[StepGraph] = None
    params.assert_empty("'tango run'")

    # Register run.
    run: "Run"
    if step_names:
        for step_name in step_names:
            assert step_name in step_graph, (
                f"You want to run a step called '{step_name}', but it cannot be found in the experiment config. "
                f"The config contains: {list(step_graph.keys())}."
            )
        sub_graph = step_graph.sub_graph(*step_names)
        if called_by_executor and name is not None:
            try:
                run = workspace.registered_run(name)
            except KeyError:
                raise RuntimeError(
                    "The CLI was called by `MulticoreExecutor.execute_step_graph`, but "
                    f"'{name}' is not already registered as a run. This should never happen!"
                )
        else:
            run = workspace.register_run((step for step in sub_graph.values()), name)
    else:
        run = workspace.register_run((step for step in step_graph.values()), name)

    def log_and_execute_run():
        if not called_by_executor:
            cli_logger.info("[green]Starting new run [bold]%s[/][/]", run.name)

        executor_output: Optional[ExecutorOutput] = None
        if step_names:
            assert sub_graph is not None
            executor_output = executor.execute_sub_graph_for_steps(
                sub_graph, *step_names, run_name=run.name
            )
        else:
            executor_output = executor.execute_step_graph(step_graph, run_name=run.name)

        if executor_output is not None and executor_output.failed:
            cli_logger.error("[red]\N{ballot x} Run [bold]%s[/] finished with errors[/]", run.name)
        elif not called_by_executor:
            cli_logger.info("[green]\N{check mark} Finished run [bold]%s[/][/]", run.name)

        if executor_output is not None:
            if not called_by_executor:
                _display_run_results(run, step_graph, workspace, executor_output)
            if executor_output.failed:
                raise CliRunError

    if called_by_executor:
        assert step_names is not None and len(step_names) == 1

        from tango.common.aliases import EnvVarNames

        # We set this environment variable so that any steps that contain multiprocessing
        # and call `initialize_worker_logging` also log the messages with the `step_name` prefix.
        os.environ[EnvVarNames.LOGGING_PREFIX.value] = f"step {step_names[0]}"
        initialize_prefix_logging(prefix=f"step {step_names[0]}", main_process=False)
        log_and_execute_run()
    else:
        # Capture logs to file.
        with workspace.capture_logs_for_run(run.name):
            log_and_execute_run()

    return run.name


def _display_run_results(
    run: "Run", step_graph: StepGraph, workspace: Workspace, executor_output: "ExecutorOutput"
) -> None:
    from rich.table import Table

    table = Table(caption_style="")
    table.add_column("Step Name", justify="left", style="cyan")
    table.add_column("Status", justify="left")
    table.add_column("Results", justify="left")
    last_cached_step: Optional[str] = None
    for step_name in sorted(step_graph):
        status_str: str
        result_str: str = "[grey62]N/A[/]"
        if step_name in executor_output.failed:
            status_str = "[red]\N{ballot x} failed[/]"
            execution_metadata = executor_output.failed[step_name]
            if execution_metadata.logs_location is not None:
                result_str = f"[cyan]{execution_metadata.logs_location}[/]"
        elif step_name in executor_output.not_run:
            status_str = "[yellow]- not run[/]"
        elif step_name in executor_output.successful:
            status_str = "[green]\N{check mark} succeeded[/]"
            execution_metadata = executor_output.successful[step_name]
            if execution_metadata.result_location is not None:
                result_str = f"[cyan]{execution_metadata.result_location}[/]"
                last_cached_step = step_name
            elif execution_metadata.logs_location is not None:
                result_str = f"[cyan]{execution_metadata.logs_location}[/]"
        else:
            continue

        table.add_row(step_name, status_str, result_str)

    caption_parts: List[str] = []
    if executor_output.failed:
        caption_parts.append(
            f"[red]\N{ballot x}[/] [italic]{len(executor_output.failed)} failed[/]"
        )
    if executor_output.successful:
        caption_parts.append(
            f"[green]\N{check mark}[/] [italic]{len(executor_output.successful)} succeeded[/]"
        )
    if executor_output.not_run:
        caption_parts.append(f"[italic]{len(executor_output.not_run)} not run[/]")
    table.caption = ", ".join(caption_parts)

    cli_logger.info(table)

    if not executor_output.failed and last_cached_step is not None:
        from rich.syntax import Syntax

        example = Syntax(
            "\n".join(
                [
                    ">>> from tango import Workspace",
                    f'>>> workspace = Workspace.from_url("{workspace.url}")',
                    f'>>> workspace.step_result_for_run("{run.name}", "{last_cached_step}")',
                ]
            ),
            "python",
        )
        cli_logger.info("Use your workspace to get the cached result of a step, e.g.")
        cli_logger.info(example)


if __name__ == "__main__":
    main()
