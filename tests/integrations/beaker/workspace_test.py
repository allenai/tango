from tango.integrations.beaker.workspace import BeakerWorkspace
from tango.step_info import StepState
from tango.workspace import Workspace
from tango.common.testing import FloatStep


def test_from_url(beaker_workspace: str):
    workspace = Workspace.from_url(f"beaker://{beaker_workspace}")
    assert isinstance(workspace, BeakerWorkspace)


def test_direct_usage(beaker_workspace: str):
    workspace = BeakerWorkspace(beaker_workspace)

    step = FloatStep(step_name="float", result=1.0)
    run = workspace.register_run([step])
    assert run.name in workspace.registered_runs()

    assert workspace.step_info(step).state == StepState.INCOMPLETE
    workspace.step_starting(step)
    assert workspace.step_info(step).state == StepState.RUNNING
    workspace.step_finished(step, 1.0)
    assert workspace.step_info(step).state == StepState.COMPLETED
    assert workspace.step_result_for_run(run.name, "float") == 1.0
