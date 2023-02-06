import os
import subprocess
from typing import Callable, ClassVar, Optional

from tango.common import PathOrStr, Registrable
from tango.step import Step


class RegistrableFunction(Registrable):
    WRAPPED_FUNC: ClassVar[Callable]

    def __call__(self, *args, **kwargs):
        return self.__class__.WRAPPED_FUNC(*args, **kwargs)


def make_registrable(name: Optional[str] = None, *, exist_ok: bool = False):
    def function_wrapper(func):
        @RegistrableFunction.register(name or func.__name__, exist_ok=exist_ok)
        class WrapperFunc(RegistrableFunction):
            WRAPPED_FUNC = func

        return WrapperFunc()

    return function_wrapper


@make_registrable(exist_ok=True)
def check_path_existence(path: PathOrStr):
    assert os.path.exists(path), f"Output not found at {path}!"


@Step.register("shell_step")
class ShellStep(Step):
    """
    This step runs a shell command, and returns the standard output as a string.

    .. tip::

        Registered as a :class:`~tango.step.Step` under the name "shell_step".

    :param shell_command: The shell command to run.
    :param output_path: The step makes no assumptions about the command being run. If your command produces some
        output, ou can optionally specify the output path, for recording the output location, and optionally
        validating it. See `validate_output` argument for this.
    :param validate_output: If an expected `output_path` has been specified, you can choose to validate that the
    step produced the correct output. By default, it will just check if the `output_path` exists, but you can
    pass any other validating function. For example, if your command is a script generating a model output,
    you can check if the model weights can be loaded.
    :param kwargs: Other kwargs to be passed to `subprocess.run()`. Note that by default we
        set `shell = True`.
    """

    def run(  # type: ignore[override]
        self,
        shell_command: str,
        output_path: Optional[PathOrStr] = None,
        validate_output: RegistrableFunction = check_path_existence,
        **kwargs,
    ):
        output = self.run_command(shell_command, **kwargs)
        self.logger.info(output)
        if output_path is not None:
            validate_output(output_path)
            self.logger.info(f"Output found at: {output_path}")

        return str(output.decode("utf-8"))

    def run_command(self, command: str, **kwargs):
        self.logger.info("Command: " + command)
        # shell=True to take advantage of environment variables, etc.
        process = subprocess.run(command, capture_output=True, shell=True, **kwargs)
        if process.returncode != 0:
            raise RuntimeError(f"The command failed with the following errors: {process.stderr}")
        return process.stdout
