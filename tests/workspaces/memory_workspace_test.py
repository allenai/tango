from tango.common.testing.steps import FloatStep
from tango.workspaces import MemoryWorkspace


def test_remove_step():
    workspace = MemoryWorkspace()
    step = FloatStep(step_name="float", result=1.0)

    workspace.step_starting(step)
    workspace.step_finished(step, 1.0)
    cache = workspace.step_cache

    assert step.unique_id in workspace.unique_id_to_info
    assert step in cache

    workspace.remove_step(step.unique_id)
    cache = workspace.step_cache

    assert step.unique_id not in workspace.unique_id_to_info
    assert step not in cache
