import os
import subprocess
from typing import Callable, Optional

from tango.common import PathOrStr
from tango.step import Step


def check_path_existence(path: PathOrStr):
    assert os.path.exists(path), f"Output not found at {path}!"


@Step.register("shell_step")
class ShellStep(Step):
    """
    The script step assumes
    """

    def run(
        self,
        shell_command: str,
        output_path: Optional[PathOrStr] = None,
        validate_output: Callable = check_path_existence,
        **kwargs,
    ):
        """
        Script needs to be in the specific format.
        """
        # output_path = output_path or self.work_dir / "output"
        output = self.run_command(shell_command, **kwargs)
        self.logger.info(output)
        if output_path is not None:
            validate_output(output_path)
        return str(output)

    def run_command(self, command: str, **kwargs):
        self.logger.info("Command: " + command)
        # shell=True to take advantage of environment variables, etc.
        process = subprocess.run(command, capture_output=True, shell=True, **kwargs)
        if process.returncode != 0:
            raise RuntimeError(f"The command failed with the following errors: {process.stderr}")
        return process.stdout
