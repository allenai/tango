import petname
import pytest
from beaker import DataMount

from tango.common.testing import TangoTestCase, run_experiment
from tango.integrations.beaker.executor import BeakerExecutor
from tango.integrations.beaker.workspace import BeakerWorkspace
from tango.settings import TangoGlobalSettings
from tango.step_info import StepState
from tango.workspaces import default_workspace


def test_from_params(beaker_workspace_name: str):
    executor = BeakerExecutor.from_params(
        dict(
            beaker_workspace=beaker_workspace_name,
            beaker_image="ai2/conda",
            github_token="FAKE_TOKEN",
            datasets=[{"source": {"beaker": "some-dataset"}, "mount_path": "/input"}],
        ),
        workspace=default_workspace,
        clusters=["fake-cluster"],
    )
    assert len(executor.datasets) == 1
    assert isinstance(executor.datasets[0], DataMount)
    assert executor.datasets[0].source.beaker == "some-dataset"


@pytest.fixture
def settings(beaker_workspace_name: str) -> TangoGlobalSettings:
    return TangoGlobalSettings(
        workspace={"type": "beaker", "beaker_workspace": beaker_workspace_name},
        executor={
            "type": "beaker",
            "beaker_workspace": beaker_workspace_name,
            "clusters": ["ai2/allennlp-cirrascale", "ai2/general-cirrascale"],
        },
    )


def test_beaker_executor(
    settings: TangoGlobalSettings, beaker_workspace_name: str, patched_unique_id_suffix: str
):
    run_name = petname.generate()
    with run_experiment(
        {"steps": {"hello": {"type": "string", "result": "Hello, World!"}}},
        settings=settings,
        workspace_url=f"beaker://{beaker_workspace_name}",
        name=run_name,
        multicore=None,
    ):
        workspace = BeakerWorkspace(beaker_workspace=beaker_workspace_name)
        run = workspace.registered_run(run_name)
        assert run.steps["hello"].state == StepState.COMPLETED
