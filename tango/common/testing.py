import logging
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

from tango.common.aliases import EnvVarNames, PathOrStr
from tango.common.logging import initialize_logging, teardown_logging
from tango.common.params import Params


class TangoTestCase:
    """
    A custom testing class that

    * disables some of the more verbose logging,
    * creates and destroys a temp directory as a test fixture, and
    * restores the internal state of the `Registrable` class at the end of each test method.

    """

    PROJECT_ROOT = (Path(__file__).parent / ".." / "..").resolve()
    """
    Root of the git repository.
    """

    MODULE_ROOT = PROJECT_ROOT / "tango"
    """
    Root of the tango module.
    """

    TESTS_ROOT = PROJECT_ROOT / "tests"
    """
    Root of the tests directory.
    """

    FIXTURES_ROOT = PROJECT_ROOT / "test_fixtures"
    """
    Root of the test fixtures directory.
    """

    def setup_method(self):
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.DEBUG
        )

        # Disabling some of the more verbose logging statements that typically aren't very helpful
        # in tests.
        logging.getLogger("urllib3.connectionpool").disabled = True

        # Create a temporary scratch directory.
        self.TEST_DIR = Path(tempfile.mkdtemp(prefix="tango_tests"))
        os.makedirs(self.TEST_DIR, exist_ok=True)

        # Set an artificial console width so logs are not mangled.
        os.environ[EnvVarNames.CONSOLE_WIDTH.value] = str(300)

    def teardown_method(self):
        shutil.rmtree(self.TEST_DIR)
        if EnvVarNames.CONSOLE_WIDTH.value in os.environ:
            del os.environ[EnvVarNames.CONSOLE_WIDTH.value]

    def run(
        self,
        config: Union[PathOrStr, Dict[str, Any], Params],
        overrides: Optional[Union[Dict[str, Any], str]] = None,
        include_package: Optional[List[str]] = None,
        workspace_url: Optional[str] = None,
        step_name: Optional[str] = None,
        parallelism: int = 1,
        multicore: bool = False,
        name: Optional[str] = None,
    ) -> Path:
        from tango.__main__ import TangoGlobalSettings, _run

        if isinstance(config, (dict, Params)):
            params = config if isinstance(config, Params) else Params(config)
            config = self.TEST_DIR / "config.json"
            params.to_file(cast(Path, config))

        if isinstance(overrides, dict):
            import json

            overrides = json.dumps(overrides)

        run_name = _run(
            TangoGlobalSettings(),
            str(config),
            workspace_url=workspace_url or "local://" + str(self.TEST_DIR / "workspace"),
            overrides=overrides,
            include_package=include_package,
            start_server=False,
            step_name=step_name,
            parallelism=parallelism,
            multicore=multicore,
            name=name,
        )

        return self.TEST_DIR / "workspace" / "runs" / run_name


@contextmanager
def run_experiment(
    config: Union[PathOrStr, Dict[str, Any], Params],
    overrides: Optional[Union[Dict[str, Any], str]] = None,
    file_friendly_logging: bool = True,
    include_package: Optional[List[str]] = None,
    parallelism: int = 1,
    multicore: bool = False,
    name: Optional[str] = None,
):
    """
    A context manager to make testing experiments easier. On ``__enter__`` it runs
    the experiment and returns the path to the run directory, a temporary directory that will be
    cleaned up on ``__exit__``.
    """
    initialize_logging(enable_cli_logs=True, file_friendly_logging=file_friendly_logging)
    test_case = TangoTestCase()
    try:
        test_case.setup_method()
        yield test_case.run(
            config,
            overrides=overrides,
            include_package=include_package,
            parallelism=parallelism,
            multicore=multicore,
            name=name,
        )
    finally:
        test_case.teardown_method()
        teardown_logging()


def requires_gpus(test_method):
    """
    Decorator to indicate that a test requires multiple GPU devices.
    """
    import pytest
    import torch

    return pytest.mark.gpu(
        pytest.mark.skipif(torch.cuda.device_count() < 2, reason="2 or more GPUs required.")(
            test_method
        )
    )
