from tango.steps.shell_step import ShellStep
from tango.common.testing import TangoTestCase


def test_shell_step():
    step = ShellStep()
    result = step.run("ls -l")
    assert isinstance(result, str)

