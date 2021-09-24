from copy import deepcopy
import logging
import os
import shutil
import tempfile
from pathlib import Path

from .registrable import Registrable


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
