from beaker import DataMount

from tango.integrations.beaker.executor import BeakerExecutor
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
