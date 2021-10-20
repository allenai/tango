from contextlib import contextmanager
from copy import deepcopy
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, cast, Union

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

    @classmethod
    def setup_class(cls):
        # During teardown we'll restore the state of `Registrable`'s internal registry
        # to make sure any registered mock test classes are removed so they don't conflict
        # with other tests.
        cls._original_registry = deepcopy(Registrable._registry)

    def teardown_method(self):
        shutil.rmtree(self.TEST_DIR)

    @classmethod
    def teardown_class(cls):
        Registrable._registry = cls._original_registry

    def run(
        self,
        config: Union[PathOrStr, Dict[str, Any]],
        overrides: Optional[Union[Dict[str, Any], str]] = None,
        include_package: Optional[List[str]] = None,
    ) -> Path:
        from .params import Params
        from tango.__main__ import _run, TangoGlobalSettings

        if isinstance(config, dict):
            params = Params(config)
            config = self.TEST_DIR / "config.json"
            params.to_file(cast(Path, config))

        if isinstance(overrides, dict):
            import json

            overrides = json.dumps(overrides)

        run_dir = self.TEST_DIR / "run"
        _run(
            TangoGlobalSettings(),
            str(config),
            directory=str(run_dir),
            overrides=overrides,
            include_package=include_package,
        )
        return run_dir


@contextmanager
def run_experiment(
    config: Union[PathOrStr, Dict[str, Any]], overrides: Optional[Union[Dict[str, Any], str]] = None
):
    """
    A context manager to make testing experiments easier. On ``__enter__`` it runs
    the experiment and returns the path to the cache directory, a temporary directory that will be
    cleaned up on ``__exit__``.
    """
    test_case = TangoTestCase()
    try:
        test_case.setup_method()
        yield test_case.run(config, overrides=overrides)
    finally:
        test_case.teardown_method()
