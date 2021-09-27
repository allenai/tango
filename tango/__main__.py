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
@click.option(
    "--file-friendly-logging",
    is_flag=True,
    help="Outputs progress bar status on separate lines and slows refresh rate",
)
def main(file_friendly_logging: bool = False):
    LOG_LEVEL = logging._nameToLevel.get(os.environ.get("TANGO_LOG_LEVEL", "INFO"), logging.INFO)
    logging.basicConfig(format="[%(asctime)s %(levelname)s %(name)s] %(message)s", level=LOG_LEVEL)
    # filelock emits too many messages, so tell it to be quiet unless it has something
    # important to say.
    logging.getLogger("filelock").setLevel(max(LOG_LEVEL, logging.WARNING))

    if file_friendly_logging:
        os.environ["FILE_FRIENDLY_LOGGING"] = "true"

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
    help="The directory in which to save the results of each step",
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
def run(
    experiment: str,
    directory: Optional[Union[str, os.PathLike]] = None,
    overrides: Optional[str] = None,
    dry_run: bool = False,
    include_package: Optional[List[str]] = None,
):
    """
    Run a tango experiment.
    """
    logger = logging.getLogger("tango")

    from pathlib import Path
    from tango import step_graph_from_params, tango_dry_run, DirectoryStepCache
    from tango.common.params import Params
    from tango.common.util import import_module_and_submodules

    if include_package is not None:
        for package_name in include_package:
            import_module_and_submodules(package_name)

    params = Params.from_file(experiment)
    step_graph = step_graph_from_params(params.pop("steps"))

    # Prepare directory.
    if directory is None:
        from tempfile import mkdtemp

        directory = mkdtemp(prefix="tango-")
        print(f"Creating temporary directory for run {directory}")

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    step_cache = DirectoryStepCache(directory / "step_cache")

    if dry_run:
        for step, cached in tango_dry_run(
            (s for s in step_graph.values() if not s.only_if_needed), step_cache
        ):
            if cached:
                print(f"Getting {step.name} from cache")
            else:
                print(f"Computing {step.name}")
    else:
        # remove symlinks to old results
        for filename in directory.glob("*"):
            if filename.is_symlink():
                relative_target = os.readlink(filename)
                if not relative_target.startswith("step_cache/"):
                    continue
                logger.info(
                    f"Removing symlink '{filename.name}' to previous result {relative_target}"
                )
                filename.unlink()

        # produce results
        for name, step in step_graph.items():
            if not step.only_if_needed:
                step.ensure_result(step_cache)

        # symlink everything that has been computed
        for name, step in step_graph.items():
            if step in step_cache:
                step_link = directory / name
                if step_link.exists():
                    step_link.unlink()
                step_link.symlink_to(
                    step_cache.path_for_step(step).relative_to(directory),
                    target_is_directory=True,
                )
                print(f'The output for "{name}" is in {step_link}.')


if __name__ == "__main__":
    main()
