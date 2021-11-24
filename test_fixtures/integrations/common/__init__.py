from tango import Step
from tango.common import DatasetDict


@Step.register("generate_data")
class GenerateData(Step):
    DETERMINISTIC = True
    CACHEABLE = False

    def run(self) -> DatasetDict:  # type: ignore[override]
        import torch

        torch.manual_seed(1)
        return DatasetDict(
            {
                "train": [{"x": torch.rand(10), "y": torch.rand(1)} for _ in range(64)],
                "validation": [{"x": torch.rand(10), "y": torch.rand(1)} for _ in range(32)],
            }
        )
