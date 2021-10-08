from contextlib import contextmanager
from copy import deepcopy
import logging
import os
import shutil
import tempfile
import typing as t
from pathlib import Path

from .registrable import Registrable
from .util import PathOrStr


class TangoTestCase:
    """
    A custom testing class that

    * disables some of the more verbose logging,
    * creates and destroys a temp directory as a test fixture, and
    * restores the internal state of the `Registrable` class at the end of each test method.

    """

    PROJECT_ROOT = (Path(__file__).parent / ".." / "..").resolve()
    MODULE_ROOT = PROJECT_ROOT / "tango"
    TESTS_ROOT = PROJECT_ROOT / "tests"
    FIXTURES_ROOT = PROJECT_ROOT / "test_fixtures"

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

        # In `teardown_method`, we'll restore the state of `Registrable`'s internal registry
        # to make sure any registered mock test classes are removed so they don't conflict
        # with other tests.
        self._original_registry = deepcopy(Registrable._registry)

    def teardown_method(self):
        shutil.rmtree(self.TEST_DIR)
        Registrable._registry = self._original_registry

    def run(self, config: t.Union[PathOrStr, t.Dict[str, t.Any]], dry_run: bool = False) -> Path:
        from .params import Params
        from tango.__main__ import _run

        if isinstance(config, dict):
            params = Params(config)
            config = self.TEST_DIR / "config.json"
            params.to_file(t.cast(Path, config))

        run_dir = self.TEST_DIR / "run"
        _run(str(config), directory=str(run_dir), dry_run=dry_run)
        return run_dir


@contextmanager
def run_experiment(config: t.Union[PathOrStr, t.Dict[str, t.Any]], dry_run: bool = False):
    """
    A context manager to make testing experiments easier. On ``__enter__`` it runs
    the experiment and returns the path to the cache directory, a temporary directory that will be
    cleaned up on ``__exit__``.
    """
    test_case = TangoTestCase()
    try:
        test_case.setup_method()
        yield test_case.run(config, dry_run=dry_run)
    finally:
        test_case.teardown_method()
