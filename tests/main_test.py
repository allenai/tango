import os
import re
import subprocess
from pathlib import Path
from typing import List, Tuple

import click
import pytest

from tango.common import Params
from tango.common.testing import TangoTestCase
from tango.local_workspace import ExecutorMetadata
from tango.version import VERSION


class TestMain(TangoTestCase):
    def clean_log_lines(
        self, log_lines: List[str], file_friendly_logging: bool = False
    ) -> List[str]:
        out = []
        for line in log_lines:
            # Remove the logging prefix with PID, timestamp, level, etc so we're just left
            # with the message.
            line = re.sub(
                r"^\[\d+ \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} (DEBUG|INFO|WARNING|ERROR) [^\]]+\] ",
                "",
                line,
            )
            unstyled_line = click.unstyle(line)
            if file_friendly_logging:
                assert line == unstyled_line
            line = unstyled_line
            if line:
                out.append(line.strip())
        return out

    def check_logs(
        self,
        run_dir: Path,
        process_result: subprocess.CompletedProcess,
        file_friendly_logging: bool = False,
    ) -> Tuple[List[str], List[str]]:
        stdout_lines = self.clean_log_lines(
            process_result.stdout.decode().replace("\r", "\n").split("\n")
        )

        log_file = run_dir / "out.log"
        assert log_file.is_file()

        log_lines = open(log_file).readlines()
        cleaned_log_lines = self.clean_log_lines(log_lines, file_friendly_logging)

        # The first few log messages in stdout may not be in the log file, since those get
        # emitted before the run dir is created.
        filtered_stdout_lines = stdout_lines[
            next(i for i, line in enumerate(stdout_lines) if line.startswith("Server started at")) :
        ]
        for line in filtered_stdout_lines:
            assert line in cleaned_log_lines

        return log_lines, cleaned_log_lines

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
        result = subprocess.run(cmd, capture_output=True)
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

        # Check for requirements.txt file.
        assert (run_dir / "hello_world" / "requirements.txt").is_file()

        # Check logs.
        self.check_logs(run_dir, result)

        # Running again shouldn't create any more directories in the cache.
        result = subprocess.run(cmd)
        assert result.returncode == 0
        assert len(os.listdir(self.TEST_DIR / "cache")) == 2
        # We should see two runs now.
        assert len(os.listdir(self.TEST_DIR / "runs")) == 2

    def test_experiment_with_memory_workspace(self):
        cmd = [
            "tango",
            "run",
            str(self.FIXTURES_ROOT / "experiment" / "hello_world.jsonnet"),
            "-i",
            "test_fixtures.package",
            "-d",
            str(self.TEST_DIR),
            "-o",
            "{'workspace':{'type':'memory'}}",
        ]
        result = subprocess.run(cmd, capture_output=True)
        assert result.returncode == 0

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

    @pytest.mark.parametrize("start_method", ["fork", "spawn"])
    @pytest.mark.parametrize("file_friendly_logging", [True, False])
    def test_experiment_with_multiprocessing(self, file_friendly_logging, start_method):
        cmd = (
            [
                "tango",
                "--log-level",
                "info",
                "--start-method",
                start_method,
            ]
            + ([] if not file_friendly_logging else ["--file-friendly-logging"])
            + [
                "run",
                str(self.FIXTURES_ROOT / "experiment" / "multiprocessing.jsonnet"),
                "-i",
                "test_fixtures.package",
                "-d",
                str(self.TEST_DIR),
            ]
        )
        result = subprocess.run(cmd, capture_output=True)
        run_dir = next((self.TEST_DIR / "runs").iterdir())
        _, clean_log_lines = self.check_logs(run_dir, result, file_friendly_logging)
        all_logs = "\n".join(clean_log_lines)
        assert "[rank 0] Hello from worker 0!" in clean_log_lines
        assert "[rank 1] Hello from worker 1!" in clean_log_lines
        assert "[rank 0] Hello from the click logger in worker 0!" in clean_log_lines
        assert "[rank 1] Hello from the click logger in worker 1!" in clean_log_lines
        # Make sure tqdm output makes it into the log file.
        assert "progress from main process: 100%" in all_logs
        assert "progress from worker: 100%" in all_logs
