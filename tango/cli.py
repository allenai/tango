import logging
import multiprocessing as mp
import os
import sys
import warnings
from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, Optional, Sequence, Union

from tango.common.exceptions import CliRunError
from tango.common.logging import (
    cli_logger,
    initialize_logging,
    initialize_prefix_logging,
    teardown_logging,
)
from tango.common.params import Params
from tango.executor import Executor
from tango.settings import TangoGlobalSettings
from tango.step_graph import StepGraph
from tango.workspace import Workspace

if TYPE_CHECKING:
    from tango.executor import ExecutorOutput
    from tango.workspace import Run


logger = logging.getLogger(__name__)


def load_settings(settings: Union[str, Params, dict, None] = None) -> TangoGlobalSettings:
    return (
        TangoGlobalSettings.from_file(settings)
        if isinstance(settings, str)
        else TangoGlobalSettings.from_params(settings)
        if isinstance(settings, (Params, dict))
        else TangoGlobalSettings.default()
    )


@contextmanager
def tango_cli(settings: Union[TangoGlobalSettings, str, Params, dict, None] = None):
    if not isinstance(settings, TangoGlobalSettings):
        settings = load_settings(settings)

    try:
        initialize_cli(settings=settings, called_by_executor=False)
        yield
    finally:
        cleanup_cli()


def initialize_cli(
    settings: Optional[TangoGlobalSettings] = None,
    called_by_executor: bool = False,
):
    if settings is None:
        settings = TangoGlobalSettings.default()

    if not sys.warnoptions:
        warnings.simplefilter("default", category=DeprecationWarning)

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

    mp.set_start_method(settings.multiprocessing_start_method)

    if not called_by_executor:
        initialize_logging(
            log_level=settings.log_level,
            file_friendly_logging=settings.file_friendly_logging,
            enable_cli_logs=True,
        )


def cleanup_cli():
    teardown_logging()


def prepare_workspace(
    settings: Optional[TangoGlobalSettings] = None,
    workspace_url: Optional[str] = None,
) -> Workspace:
    from tango.workspaces import default_workspace

    if settings is None:
        settings = TangoGlobalSettings.default()

    workspace: Workspace
    if workspace_url is not None:
        workspace = Workspace.from_url(workspace_url)
    elif settings.workspace is not None:
        workspace = Workspace.from_params(settings.workspace)
    else:
        workspace = default_workspace

    return workspace


def prepare_executor(
    workspace: Workspace,
    settings: Optional[TangoGlobalSettings] = None,
    include_package: Optional[Sequence[str]] = None,
    parallelism: Optional[int] = None,
    multicore: Optional[bool] = None,
    called_by_executor: bool = False,
) -> Executor:
    from tango.executors import MulticoreExecutor
    from tango.workspaces import MemoryWorkspace

    if settings is None:
        settings = TangoGlobalSettings.default()

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

    return executor


def execute_step_graph(
    step_graph: StepGraph,
    workspace: Optional[Workspace] = None,
    executor: Optional[Executor] = None,
    name: Optional[str] = None,
    called_by_executor: bool = False,
    step_names: Optional[Sequence[str]] = None,
) -> str:

    if workspace is None:
        workspace = prepare_workspace()

    if executor is None:
        executor = prepare_executor(workspace=workspace)

    # Register run.
    run: "Run"
    if called_by_executor and name is not None:
        try:
            run = workspace.registered_run(name)
        except KeyError:
            raise RuntimeError(
                "The CLI was called by `MulticoreExecutor.execute_step_graph`, but "
                f"'{name}' is not already registered as a run. This should never happen!"
            )
    else:
        run = workspace.register_run((step for step in step_graph.values()), name)

    if called_by_executor:
        assert step_names is not None and len(step_names) == 1
        from tango.common.aliases import EnvVarNames

        # We set this environment variable so that any steps that contain multiprocessing
        # and call `initialize_worker_logging` also log the messages with the `step_name` prefix.
        os.environ[EnvVarNames.LOGGING_PREFIX.value] = f"step {step_names[0]}"
        initialize_prefix_logging(prefix=f"step {step_names[0]}", main_process=False)

    # Capture logs to file.
    with workspace.capture_logs_for_run(run.name) if not called_by_executor else nullcontext():
        if not called_by_executor:
            cli_logger.info("[green]Starting new run [bold]%s[/][/]", run.name)

        executor_output: ExecutorOutput = executor.execute_step_graph(step_graph, run_name=run.name)

        if executor_output.failed:
            cli_logger.error("[red]\N{ballot x} Run [bold]%s[/] finished with errors[/]", run.name)
        elif not called_by_executor:
            cli_logger.info("[green]\N{check mark} Finished run [bold]%s[/][/]", run.name)

        if executor_output is not None:
            if not called_by_executor:
                executor_output.display()
            if executor_output.failed:
                raise CliRunError

    return run.name
