import torch
from torch.utils.data import IterableDataset

from tango import Step
from tango.common import DatasetDict, IterableDatasetDict


@Step.register("random_data")
class GenerateData(Step):
    DETERMINISTIC = True
    CACHEABLE = False

    def run(self) -> DatasetDict:  # type: ignore[override]
        torch.manual_seed(1)
        return DatasetDict(
            {
                "train": [{"x": torch.rand(10), "y": torch.rand(1)} for _ in range(64)],
                "validation": [{"x": torch.rand(10), "y": torch.rand(1)} for _ in range(32)],
            }
        )


class RandomIterableDataset(IterableDataset):
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        return iter(self.data)


@Step.register("generate_streaming_data")
class GenerateStreamingData(Step):
    DETERMINISTIC = True
    CACHEABLE = False

    def run(self) -> IterableDatasetDict:  # type: ignore[override]
        torch.manual_seed(1)
        return IterableDatasetDict(
            {
                "train": RandomIterableDataset(
                    [{"x": torch.rand(10), "y": torch.rand(1)} for _ in range(64)]
                ),
                "validation": RandomIterableDataset(
                    [{"x": torch.rand(10), "y": torch.rand(1)} for _ in range(32)]
                ),
            }
        )
