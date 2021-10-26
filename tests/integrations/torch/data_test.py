import torch

from tango.integrations.torch.data import DataLoader, Sampler


def test_dataloader_from_params():
    DataLoader.from_params(
        {
            "dataset": list(range(10)),
            "batch_size": 2,
            "shuffle": True,
        }
    )


def test_samplers_registered():
    assert "torch::SequentialSampler" in Sampler.list_available()


def test_dataloader_from_params_with_sampler():
    dataloader = DataLoader.from_params(
        {
            "dataset": list(range(10)),
            "sampler": {
                "type": "torch::RandomSampler",
                "replacement": True,
            },
        }
    )
    assert isinstance(dataloader.sampler, torch.utils.data.RandomSampler)
    assert dataloader.sampler.replacement


def test_dataloader_from_params_with_batch_sampler():
    dataloader = DataLoader.from_params(
        {
            "dataset": list(range(10)),
            "sampler": {
                "type": "BatchSampler",
                "sampler": {
                    "type": "torch::RandomSampler",
                },
                "batch_size": 2,
                "drop_last": True,
            },
        }
    )
    assert isinstance(dataloader.sampler, torch.utils.data.BatchSampler)
