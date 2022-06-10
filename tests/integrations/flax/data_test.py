from tango.integrations.flax import DataLoader


def test_DataLoader() -> None:
    assert "flax::numpy_dataloader" in DataLoader.list_available()
    assert "flax::dataset_dataloader" in DataLoader.list_available()
