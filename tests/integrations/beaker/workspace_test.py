import pytest
from beaker import DatasetNotFound

from tango.common.testing.steps import FloatStep
from tango.integrations.beaker.workspace import BeakerWorkspace
from tango.step_info import StepState
from tango.workspace import Workspace


def test_from_url(beaker_workspace: str):
    print(beaker_workspace)
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


def test_remove_step(beaker_workspace: str):
    beaker_workspace = "ai2/tango_remove_cache_test"
    workspace = BeakerWorkspace(beaker_workspace)
    step = FloatStep(step_name="float", result=1.0)

    workspace.step_starting(step)
    workspace.step_finished(step, 1.0)

    step_info = workspace.step_info(step)
    dataset_name = workspace.Constants.step_artifact_name(step_info)
    cache = workspace.step_cache

    assert workspace.beaker.dataset.get(dataset_name) is not None
    assert step in cache

    workspace.remove_step(step.unique_id)
    cache = workspace.step_cache
    dataset_name = workspace.Constants.step_artifact_name(step_info)

    with pytest.raises(DatasetNotFound):
        workspace.beaker.dataset.get(dataset_name)
    assert step not in cache
