import logging
import os
from typing import Optional, Union, List

import click
from click_help_colors import HelpColorsCommand, HelpColorsGroup

from .version import VERSION


@click.group(
    cls=HelpColorsGroup,
    help_options_color="green",
    help_headers_color="yellow",
    context_settings={"max_content_width": 115},
)
@click.version_option(version=VERSION)
def main():
    LOG_LEVEL = logging._nameToLevel.get(os.environ.get("TANGO_LOG_LEVEL", "INFO"), logging.INFO)
    logging.basicConfig(format="[%(asctime)s %(levelname)s %(name)s] %(message)s", level=LOG_LEVEL)
    # filelock emits too many messages, so tell it to be quiet unless it has something
    # important to say.
    logging.getLogger("filelock").setLevel(max(LOG_LEVEL, logging.WARNING))

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
    "--dry-run",
    is_flag=True,
    help="Only show what tango would run without actually running any steps.",
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
    dry_run: bool = False,
    include_package: Optional[List[str]] = None,
    file_friendly_logging: bool = False,
):
    """
    Run a tango experiment.
    """
    if file_friendly_logging:
        os.environ["FILE_FRIENDLY_LOGGING"] = "true"

    from pathlib import Path
    from tango import (
        step_graph_from_params,
        tango_dry_run,
        StepCache,
        DirectoryStepCache,
        MemoryStepCache,
    )
    from tango.common.file_lock import FileLock
    from tango.common.params import Params
    from tango.common.util import import_module_and_submodules

    logger = logging.getLogger("tango")

    # Import included packages to find registered components.
    if include_package is not None:
        for package_name in include_package:
            import_module_and_submodules(package_name)

    # Read params and create step graph.
    params = Params.from_file(experiment, params_overrides=overrides or "")
    step_graph = step_graph_from_params(params.pop("steps"))

    # Prepare directory.
    if directory is None and not dry_run:
        from tempfile import mkdtemp

        directory = mkdtemp(prefix="tango-")
        click.echo(
            "Creating temporary directory for run: " + click.style(f"{directory}", fg="yellow")
        )

    # Initialize step cache.
    step_cache: StepCache
    if directory:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        step_cache = DirectoryStepCache(directory / "step_cache")
    else:
        step_cache = MemoryStepCache()

    if dry_run:
        click.secho("Dry run:", bold=True)
        for name, step in step_graph.items():
            if step.only_if_needed:
                click.secho(f"Skipping {name} as it is not needed", fg="yellow")
        dry_run_steps = tango_dry_run(
            (s for s in step_graph.values() if not s.only_if_needed), step_cache
        )
        for i, (step, cached) in enumerate(dry_run_steps):
            if cached:
                click.echo(
                    f"[{i+1}/{len(dry_run_steps)}] "
                    + click.style("✓ Getting ", fg="blue")
                    + click.style(f"{step.name} ", bold=True, fg="blue")
                    + click.style("from cache", fg="blue")
                )
            else:
                click.echo(
                    f"[{i+1}/{len(dry_run_steps)}] "
                    + click.style("● Computing ", fg="green")
                    + click.style(f"{step.name}", bold=True, fg="green")
                )
    else:
        assert isinstance(directory, Path)
        assert isinstance(step_cache, DirectoryStepCache)

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

            # Produce results.
            for name, step in step_graph.items():
                if not step.only_if_needed:
                    step.ensure_result(step_cache)

            # Symlink everything that has been computed.
            for name, step in step_graph.items():
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
                        + click.style(f"{step_link}.", bold=True, fg="green")
                    )
        finally:
            directory_lock.release()


if __name__ == "__main__":
    main()
