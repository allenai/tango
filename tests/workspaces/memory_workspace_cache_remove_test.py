# import pytest

from tango import Step
from tango.step_info import StepState
from tango.workspaces import MemoryWorkspace


class AdditionStep(Step):
    def run(self, a: int, b: int) -> int:
        return a + b


def test_step_cache_remove():
    workspace = MemoryWorkspace()
    step1 = AdditionStep(a=1, b=2)
    step_info = workspace.step_info(step1)
    assert step_info.state == StepState.INCOMPLETE
    result1 = step1.result(workspace)
    step_info = workspace.step_info(step1)
    assert step_info.state == StepState.COMPLETED
    step1_unique_id = step1.unique_id
    workspace.step_cache_remove(step1_unique_id)


