from tango.integrations.flax import DataLoader


def test_dataloader() -> None:
    assert "flax::dataloader" in DataLoader.list_available()