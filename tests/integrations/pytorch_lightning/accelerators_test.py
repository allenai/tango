from tango.integrations.pytorch_lightning.accelerators import LightningAccelerator


def test_all_accelerators_registered():
    assert "pytorch_lightning::GPUAccelerator" in LightningAccelerator.list_available()
