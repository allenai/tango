import subprocess
from pathlib import Path

import petname
import pytest
from beaker import DataMount

from tango.common.testing import TangoTestCase
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
def settings_path(tmp_path: Path, beaker_workspace: str) -> Path:
    settings = TangoGlobalSettings(
        workspace={"type": "beaker", "beaker_workspace": beaker_workspace},
        executor={
            "type": "beaker",
            "beaker_workspace": beaker_workspace,
            "clusters": ["ai2/allennlp-cirrascale", "ai2/general-cirrascale"],
        },
    )
    path = tmp_path / "tango.yml"
    settings.to_file(path)
    return path


def test_beaker_executor(settings_path: Path, beaker_workspace: str):
    run_name = petname.generate()
    cmd = [
        "tango",
        "--settings",
        str(settings_path),
        "--log-level",
        "info",
        "run",
        str(TangoTestCase.FIXTURES_ROOT / "experiment" / "hello_world.jsonnet"),
        "--name",
        run_name,
        "--allow-dirty",
    ]
    result = subprocess.run(cmd, capture_output=True)
    assert result.returncode == 0, result.stderr.decode().replace("\r", "\n")

    workspace = BeakerWorkspace(beaker_workspace=beaker_workspace)
    run = workspace.registered_run(run_name)
    assert run.steps["hello_world"].state == StepState.COMPLETED
