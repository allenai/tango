import logging
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

from tango.common.aliases import PathOrStr
from tango.common.logging import initialize_logging


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

    def teardown_method(self):
        shutil.rmtree(self.TEST_DIR)

    def run(
        self,
        config: Union[PathOrStr, Dict[str, Any]],
        overrides: Optional[Union[Dict[str, Any], str]] = None,
        include_package: Optional[List[str]] = None,
    ) -> Path:
        from tango.__main__ import TangoGlobalSettings, _run

        from .params import Params

        if isinstance(config, dict):
            params = Params(config)
            config = self.TEST_DIR / "config.json"
            params.to_file(cast(Path, config))

        if isinstance(overrides, dict):
            import json

            overrides = json.dumps(overrides)

        return _run(
            TangoGlobalSettings(),
            str(config),
            workspace_dir=str(self.TEST_DIR / "workspace"),
            overrides=overrides,
            include_package=include_package,
            start_server=False,
        )


@contextmanager
def run_experiment(
    config: Union[PathOrStr, Dict[str, Any]], overrides: Optional[Union[Dict[str, Any], str]] = None
):
    """
    A context manager to make testing experiments easier. On ``__enter__`` it runs
    the experiment and returns the path to the cache directory, a temporary directory that will be
    cleaned up on ``__exit__``.
    """
    initialize_logging(enable_click_logs=True)
    test_case = TangoTestCase()
    try:
        test_case.setup_method()
        yield test_case.run(config, overrides=overrides)
    finally:
        test_case.teardown_method()
