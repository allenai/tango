from tango.integrations.pytorch_lightning.loggers import LightningLogger


def test_all_loggers_registered():
    assert "pytorch_lightning::CSVLogger" in LightningLogger.list_available()
