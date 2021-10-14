import logging
import os
from typing import Optional, Union, List

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
    include_package: Optional[List[str]] = None,
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


def _run(
    experiment: str,
    directory: Optional[Union[str, os.PathLike]] = None,
    overrides: Optional[str] = None,
    include_package: Optional[List[str]] = None,
    file_friendly_logging: bool = False,
):
    if file_friendly_logging:
        os.environ["FILE_FRIENDLY_LOGGING"] = "true"

    from pathlib import Path
    from tango.common.file_lock import FileLock
    from tango.common.params import Params
    from tango.common.util import import_module_and_submodules
    from tango.executor import Executor
    from tango.step_cache import StepCache
    from tango.step_graph import StepGraph

    logger = logging.getLogger("tango")

    # Import included packages to find registered components.
    if include_package is not None:
        for package_name in include_package:
            import_module_and_submodules(package_name)

    # Read params.
    params = Params.from_file(experiment, params_overrides=overrides or "")

    # Prepare directory.
    if directory is None:
        from tempfile import mkdtemp

        directory = mkdtemp(prefix="tango-")
        click.echo(
            "Creating temporary directory for run: " + click.style(f"{directory}", fg="yellow")
        )
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    # Initialize step graph, executor, and cache.
    step_graph = StepGraph(params.pop("steps", keep_as_dict=True))
    executor = Executor.from_params(
        params.pop("executor", default={}), include_package=include_package
    )
    step_cache = StepCache.from_params(
        params.pop("cache", default={}), dir=directory / "step_cache"
    )

    # Acquire lock on directory.
    directory_lock = FileLock(directory / ".lock", read_only_ok=True)
    directory_lock.acquire_with_updates(desc="acquiring directory lock...")

    try:
        # Remove symlinks to old results.
        for filename in directory.glob("*"):
            if filename.is_symlink():
                relative_target = os.readlink(filename)
                if not relative_target.startswith("step_cache/"):
                    continue
                logger.debug(
                    f"Removing symlink '{filename.name}' to previous result {relative_target}"
                )
                filename.unlink()

        # Produce results and symlink everything that has been computed.
        for step in executor.execute_step_graph(step_graph, step_cache):
            name = step.name
            if step in step_cache:
                step_link = directory / name
                if step_link.exists():
                    step_link.unlink()
                step_link.symlink_to(
                    step_cache.path_for_step(step).relative_to(directory),
                    target_is_directory=True,
                )
                click.echo(
                    click.style("✓ The output for ", fg="green")
                    + click.style(f'"{name}"', bold=True, fg="green")
                    + click.style(" is in ", fg="green")
                    + click.style(f"{step_link}", bold=True, fg="green")
                )
    finally:
        directory_lock.release()


if __name__ == "__main__":
    main()
