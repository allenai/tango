import petname
import pytest
from beaker import DataMount

from tango.common.exceptions import ConfigurationError
from tango.common.testing import run_experiment
from tango.executor import Executor
from tango.integrations.beaker.executor import BeakerExecutor
from tango.integrations.beaker.workspace import BeakerWorkspace
from tango.settings import TangoGlobalSettings
from tango.workspaces import default_workspace


def test_from_params(beaker_workspace_name: str):
    executor = Executor.from_params(
        dict(
            type="beaker",
            beaker_workspace=beaker_workspace_name,
            beaker_image="ai2/conda",
            github_token="FAKE_TOKEN",
            datasets=[{"source": {"beaker": "some-dataset"}, "mount_path": "/input"}],
            budget="ai2/allennlp",
        ),
        workspace=BeakerWorkspace(workspace=beaker_workspace_name),
        clusters=["fake-cluster"],
    )
    assert isinstance(executor, BeakerExecutor)
    assert executor.datasets is not None
    assert len(executor.datasets) == 1
    assert isinstance(executor.datasets[0], DataMount)
    assert executor.datasets[0].source.beaker == "some-dataset"


def test_init_with_mem_workspace(beaker_workspace_name: str):
    with pytest.raises(ConfigurationError, match="MemoryWorkspace"):
        BeakerExecutor(
            workspace=default_workspace,
            beaker_workspace=beaker_workspace_name,
            beaker_image="ai2/conda",
            github_token="FAKE_TOKEN",
            clusters=["fake-cluster"],
            budget="ai2/allennlp",
        )


@pytest.fixture
def settings(beaker_workspace_name: str) -> TangoGlobalSettings:
    return TangoGlobalSettings(
        workspace={"type": "beaker", "beaker_workspace": beaker_workspace_name},
        executor={
            "type": "beaker",
            "beaker_workspace": beaker_workspace_name,
            "install_cmd": "pip install .[beaker]",
            "clusters": ["ai2/allennlp-cirrascale", "ai2/general-cirrascale"],
            "budget": "ai2/allennlp",
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
        workspace = BeakerWorkspace(workspace=beaker_workspace_name)
        assert "hello" in workspace.registered_run(run_name).steps
