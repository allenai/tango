from tango.integrations.pytorch_lightning.callbacks import LightningCallback


def test_all_callbacks_registered():
    assert "pytorch_lightning::ModelCheckpoint" in LightningCallback.list_available()
