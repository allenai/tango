from tango.integrations.pytorch_lightning.profilers import LightningProfiler


def test_all_profilers_registered():
    assert "pytorch_lightning::SimpleProfiler" in LightningProfiler.list_available()
