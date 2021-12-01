import os
import subprocess
from pathlib import Path

from tango.common import Params
from tango.common.testing import TangoTestCase
from tango.local_workspace import ExecutorMetadata
from tango.version import VERSION


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
        assert len(os.listdir(self.TEST_DIR / "cache")) == 2
        run_dir = next((self.TEST_DIR / "runs").iterdir())
        assert (run_dir / "hello").is_dir()
        assert (run_dir / "hello" / "cache-metadata.json").is_file()
        assert (run_dir / "hello" / "executor-metadata.json").is_file()
        assert (run_dir / "hello_world").is_dir()

        # Check metadata.
        metadata_path = run_dir / "hello_world" / "executor-metadata.json"
        assert metadata_path.is_file()
        metadata_params = Params.from_file(metadata_path)
        metadata = ExecutorMetadata.from_params(metadata_params)
        assert metadata.config == {
            "type": "concat_strings",
            "string1": {"type": "ref", "ref": "StringStep-4cHbmoHigd3rvNn3w7shc1d45WA1ijSp"},
            "string2": "World!",
            "join_with": ", ",
        }
        if (Path.cwd() / ".git").exists():
            assert metadata.git.commit is not None
            assert metadata.git.remote is not None

        assert (run_dir / "hello_world" / "requirements.txt").is_file()

        # Running again shouldn't create any more directories in the cache.
        result = subprocess.run(cmd)
        assert result.returncode == 0
        assert len(os.listdir(self.TEST_DIR / "cache")) == 2
        # We should see two runs now.
        assert len(os.listdir(self.TEST_DIR / "runs")) == 2

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
