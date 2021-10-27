import os
import subprocess

from tango.common import Params
from tango.common.testing import TangoTestCase
from tango.executor import ExecutorMetadata
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
        assert len(os.listdir(self.TEST_DIR / "step_cache")) == 2
        assert (self.TEST_DIR / "hello").is_dir()
        assert (self.TEST_DIR / "hello" / "cache-metadata.json").is_file()
        assert (self.TEST_DIR / "hello" / "executor-metadata.json").is_file()
        assert (self.TEST_DIR / "hello_world").is_dir()

        # Check metadata.
        metadata_path = self.TEST_DIR / "hello_world" / "executor-metadata.json"
        assert metadata_path.is_file()
        metadata_params = Params.from_file(metadata_path)
        metadata = ExecutorMetadata.from_params(metadata_params)
        assert metadata.config == {
            "type": "concat_strings",
            "string1": {"type": "ref", "ref": "hello"},
            "string2": "World!",
            "join_with": ", ",
        }
        assert metadata.git.commit is not None
        assert metadata.git.remote is not None

        assert (self.TEST_DIR / "hello_world" / "requirements.txt").is_file()

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
