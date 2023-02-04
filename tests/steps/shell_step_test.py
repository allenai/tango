import pytest

from tango.common.testing import TangoTestCase
from tango.steps.shell_step import ShellStep, make_registrable


class TestShellStep(TangoTestCase):
    def test_shell_step(self):
        step = ShellStep()
        result = step.run("echo hello")
        assert isinstance(result, str)
        assert result == "hello\n"

    def test_shell_step_failure(self):
        step = ShellStep()
        with pytest.raises(RuntimeError):
            step.run("ls -l non_existent_path")

    def test_shell_step_with_output_path(self, caplog):
        output_path = self.TEST_DIR / "test-folder"
        step = ShellStep()
        step.run(f"mkdir {output_path}", output_path=output_path)
        assert f"Output found at: {output_path}" in caplog.text

    def test_shell_step_different_validation(self, caplog):
        @make_registrable(exist_ok=True)
        def validate_func(path):
            """
            Validates that the file contents of the `path` are a json string.
            """
            import json

            with open(path) as f:
                json.load(f)

        output_path = self.TEST_DIR / "hello.json"
        command = f"python3 -c \"import json; print(json.dumps({{'a': 23}}))\" > {output_path}"
        step = ShellStep()
        step.run(command, output_path=output_path, validate_output=validate_func)
        assert f"Output found at: {output_path}" in caplog.text

    def test_shell_step_in_config(self, caplog):
        output_path = str(self.TEST_DIR / "test-folder")
        config = {
            "steps": {
                "create_dir": {
                    "type": "shell_step",
                    "shell_command": f"mkdir {output_path}",
                    "output_path": output_path,
                    "validate_output": {"type": "check_path_existence"},
                },
            }
        }

        # Regular run contains all step outputs.
        self.run(config)
        assert f"Output found at: {output_path}" in caplog.text
