import json
import os
import re
import subprocess
from pathlib import Path
from typing import List, Tuple

import click
import pytest

from tango.common.testing import TangoTestCase
from tango.settings import TangoGlobalSettings
from tango.version import VERSION


class TestRun(TangoTestCase):
    def clean_log_lines(
        self, log_lines: List[str], file_friendly_logging: bool = False
    ) -> List[str]:
        out = []
        for line in log_lines:
            unstyled_line = click.unstyle(line)
            if file_friendly_logging:
                assert line == unstyled_line
            line = unstyled_line
            parts = re.split(r"(DEBUG|INFO|WARNING|ERROR|CRITICAL)\s+", line)
            if len(parts) >= 3:
                line = "".join(parts[2:])
            elif len(parts) == 1:
                line = parts[0]
            else:
                raise ValueError(str(parts))
            line = re.sub(r"\s+[^ ]+$", "", line)
            if line:
                out.append(line.strip())
        return out

    def check_logs(
        self,
        run_dir: Path,
        process_result: subprocess.CompletedProcess,
        file_friendly_logging: bool = False,
    ) -> Tuple[List[str], List[str]]:
        stdout_lines = process_result.stdout.decode().replace("\r", "\n").split("\n")
        cleaned_stdout_lines = self.clean_log_lines(stdout_lines, file_friendly_logging)

        log_file = run_dir / "out.log"
        assert log_file.is_file()

        log_lines = open(log_file).read().split("\n")
        cleaned_log_lines = self.clean_log_lines(log_lines)

        for line in cleaned_stdout_lines[
            next(i for i, line in enumerate(stdout_lines) if "Starting new run" in line) :
        ]:
            assert line in cleaned_log_lines

        return log_lines, cleaned_log_lines

    def test_version(self):
        result = subprocess.run(["tango", "--version"], capture_output=True, text=True)
        assert result.returncode == 0
        assert VERSION in result.stdout

    @pytest.mark.parametrize("log_level", ["debug", "info", "warning", "error"])
    @pytest.mark.parametrize("raise_error", (True, False))
    def test_logging_all_levels(self, log_level: str, raise_error):
        cmd = [
            "tango",
            "--log-level",
            log_level,
            "run",
            str(self.FIXTURES_ROOT / "experiment" / "noisy.jsonnet"),
            "-w",
            str(self.TEST_DIR),
            "-o",
            json.dumps({"steps.noisy_step.raise_error": raise_error}),
        ]
        result = subprocess.run(cmd, capture_output=True)
        run_dir = next((self.TEST_DIR / "runs").iterdir())
        if raise_error:
            assert result.returncode == 1
        else:
            assert result.returncode == 0
        _, cleaned_log_lines = self.check_logs(run_dir, result)

        # Debug messages.
        assert cleaned_log_lines.count("debug message from cli_logger") == 1
        assert cleaned_log_lines.count("debug message") == (1 if log_level == "debug" else 0)

        # Info messages.
        assert cleaned_log_lines.count("info message from cli_logger") == 1
        assert cleaned_log_lines.count("info message") == (
            1 if log_level in {"debug", "info"} else 0
        )

        # Warning messages.
        assert cleaned_log_lines.count("warning message from cli_logger") == 1
        assert cleaned_log_lines.count("warning message") == (
            1 if log_level in {"debug", "info", "warning"} else 0
        )

        # Error messages.
        assert cleaned_log_lines.count("error message from cli_logger") == 1
        assert cleaned_log_lines.count("error message") == (
            1 if log_level in {"debug", "info", "warning", "error"} else 0
        )

        # Traceback.
        if raise_error:
            assert "Traceback (most recent call last):" in cleaned_log_lines
            assert "ValueError: Oh no!" in cleaned_log_lines

    def test_deterministic_experiment(self):
        cmd = [
            "tango",
            "run",
            str(self.FIXTURES_ROOT / "experiment" / "hello_world.jsonnet"),
            "-w",
            str(self.TEST_DIR),
        ]
        result = subprocess.run(cmd, capture_output=True)
        assert result.returncode == 0
        assert len(os.listdir(self.TEST_DIR / "cache")) == 2
        run_dir = next((self.TEST_DIR / "runs").iterdir())
        assert (run_dir / "hello").is_dir()
        assert (run_dir / "hello" / "cache-metadata.json").is_file()
        assert (run_dir / "hello_world").is_dir()

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
            "-w",
            "memory://",
        ]
        result = subprocess.run(cmd, capture_output=True)
        assert result.returncode == 0

    def test_experiment_with_default_workspace(self):
        cmd = [
            "tango",
            "run",
            str(self.FIXTURES_ROOT / "experiment" / "hello_world.jsonnet"),
        ]
        result = subprocess.run(cmd, capture_output=True)
        assert result.returncode == 0

    def test_random_experiment(self):
        cmd = [
            "tango",
            "run",
            str(self.FIXTURES_ROOT / "experiment" / "random.jsonnet"),
            "-w",
            str(self.TEST_DIR),
        ]
        result = subprocess.run(cmd)
        assert result.returncode == 0

    def test_run_name(self):
        name = "unique-tango-run-name"
        cmd = [
            "tango",
            "run",
            str(self.FIXTURES_ROOT / "experiment" / "hello_world.jsonnet"),
            "-w",
            str(self.TEST_DIR),
            "--name",
            name,
        ]
        result = subprocess.run(cmd, capture_output=True)
        run_dir = next((self.TEST_DIR / "runs").iterdir())
        _, clean_log_lines = self.check_logs(run_dir, result)
        assert result.returncode == 0
        assert f"Starting new run {name}" == clean_log_lines[0]

    @pytest.mark.parametrize("parallelism", [1, 2])
    @pytest.mark.parametrize("start_method", ["fork", "spawn"])
    @pytest.mark.parametrize("file_friendly_logging", [True, False])
    def test_experiment_with_logging_and_multiprocessing(
        self, parallelism, start_method, file_friendly_logging
    ):
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
                str(self.FIXTURES_ROOT / "experiment" / "logging_check.jsonnet"),
                "-w",
                str(self.TEST_DIR),
                "-j",
                str(parallelism),
            ]
        )
        result = subprocess.run(cmd, capture_output=True)
        run_dir = next((self.TEST_DIR / "runs").iterdir())
        _, clean_log_lines = self.check_logs(run_dir, result, file_friendly_logging)
        all_logs = "\n".join(clean_log_lines)
        assert "[step stringA] 0 - This is a logging test." in clean_log_lines
        assert "[step stringC] 0 - This is also a logging test." in clean_log_lines
        assert (
            "[step final_string] 0 - This is a logging test. This is being logged."
            in clean_log_lines
        )
        # Make sure tqdm output makes it into the log file.
        assert "[step stringA] log progress: 100%" in all_logs
        assert "[step stringC] log progress: 100%" in all_logs
        assert "[step final_string] log progress: 100%" in all_logs

        # And logs from steps that contain multiprocessing themselves.
        assert "[step multiprocessing_result rank 0] Hello from worker 0!" in all_logs
        assert "[step multiprocessing_result rank 1] Hello from worker 1!" in all_logs
        assert (
            "[step multiprocessing_result rank 0] Hello from the cli logger in worker 0!"
            in all_logs
        )
        assert (
            "[step multiprocessing_result rank 1] Hello from the cli logger in worker 1!"
            in all_logs
        )

        assert "[step multiprocessing_result] progress from main process: 100%" in all_logs


class TestSettings(TangoTestCase):
    def setup_method(self):
        super().setup_method()
        self._wd_backup = os.getcwd()
        os.chdir(self.TEST_DIR)
        cmd = "tango settings init -p ./tango.yml".split(" ")
        subprocess.run(cmd, check=True)

    def teardown_method(self):
        os.chdir(self._wd_backup)
        super().teardown_method()

    @property
    def settings(self) -> TangoGlobalSettings:
        return TangoGlobalSettings.from_file(self.TEST_DIR / "tango.yml")

    def test_settings_set_workspace(self):
        cmd = "tango settings set workspace ./workspace".split(" ")
        subprocess.run(cmd, check=True)
        assert self.settings.workspace == {
            "type": "local",
            "dir": str((self.TEST_DIR / "workspace").resolve()),
        }

    def test_settings_set_include_package(self):
        cmd = "tango settings set include-package tango.steps".split(" ")
        subprocess.run(cmd, check=True)
        assert self.settings.include_package == ["tango.steps"]

    def test_settings_set_include_package_invalid(self):
        cmd = "tango settings set include-package foo".split(" ")
        with pytest.raises(subprocess.CalledProcessError):
            subprocess.run(cmd, check=True)

    def test_settings_set_environment(self):
        cmd = "tango settings set env FOO BAR".split(" ")
        subprocess.run(cmd, check=True)
        assert self.settings.environment == {"FOO": "BAR"}

    def test_settings_set_environment_blocked_var(self):
        cmd = "tango settings set env TANGO_LOG_LEVEL info".split(" ")
        with pytest.raises(subprocess.CalledProcessError):
            subprocess.run(cmd, check=True)
