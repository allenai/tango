from tango.integrations.torch.data import DataLoader


def test_dataloader_from_params():
    DataLoader.from_params(
        {
            "dataset": list(range(10)),
            "batch_size": 2,
            "shuffle": True,
        }
    )
