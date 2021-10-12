import os
import subprocess

from tango.version import VERSION
from tango.common.testing import TangoTestCase


class TestMain(TangoTestCase):
    def test_version(self):
        result = subprocess.run(["tango", "--version"], capture_output=True, text=True)
        assert result.returncode == 0
        assert VERSION in result.stdout

    def test_deterministic_experiment(self):
        cmd = [
            "tango",
            "run",
            str(self.FIXTURES_ROOT / "experiment" / "hello_world.jsonnet"),
            "-i",
            "test_fixtures.package",
            "-d",
            str(self.TEST_DIR),
        ]
        result = subprocess.run(cmd)
        assert result.returncode == 0
        assert len(os.listdir(self.TEST_DIR / "step_cache")) == 2

        # Running again shouldn't create any more directories.
        result = subprocess.run(cmd)
        assert result.returncode == 0
        assert len(os.listdir(self.TEST_DIR / "step_cache")) == 2

    def test_random_experiment(self):
        cmd = [
            "tango",
            "run",
            str(self.FIXTURES_ROOT / "experiment" / "random.jsonnet"),
            "-i",
            "test_fixtures.package",
            "-d",
            str(self.TEST_DIR),
        ]
        result = subprocess.run(cmd)
        assert result.returncode == 0
