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

        # Dry run should also report that we get the results from the cache.
        cmd.append("--dry-run")
        dry_run_result = subprocess.run(cmd, capture_output=True, text=True)
        assert dry_run_result.returncode == 0
        assert "Getting hello from cache" in dry_run_result.stdout, dry_run_result.stdout
        assert "Getting hello_world from cache" in dry_run_result.stdout, dry_run_result.stdout

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

        # Dry run should report that one of the steps can be retrieved from the cache,
        # but the rest need to be computed again.
        cmd.append("--dry-run")
        dry_run_result = subprocess.run(cmd, capture_output=True, text=True)
        assert dry_run_result.returncode == 0
        assert "Getting string2 from cache" in dry_run_result.stdout
        assert "Computing string1" in dry_run_result.stdout
        assert "Computing final_string" in dry_run_result.stdout
