import pytest

from tango import Step
from tango.common.testing import TangoTestCase
from tango.step_info import StepState
from tango.workspaces import LocalWorkspace


class AdditionStep(Step):
    def run(self, a: int, b: int) -> int:  # type: ignore
        return a + b


class Test_Cache_Remove_Workspace(TangoTestCase):
    def test_step_cache_remove(self):
        workspace = LocalWorkspace(self.TEST_DIR)
        step = AdditionStep(a=1, b=2)
        step_info = workspace.step_info(step)
        assert step_info.state == StepState.INCOMPLETE
        result = step.result(workspace)
        step_info = workspace.step_info(step)
        assert step_info.state == StepState.COMPLETED
        step_unique_id = step.unique_id
        workspace.step_cache_remove(step_unique_id)
