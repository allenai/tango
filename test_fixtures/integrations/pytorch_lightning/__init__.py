import torch
import torch.nn as nn
import pytorch_lightning as pl

from tango.common.dataset_dict import DatasetDict
from tango.step import Step
from tango.integrations.pytorch_lightning import LightningModule


@LightningModule.register("basic_regression")
class BasicRegression(LightningModule):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()
        self.mse = nn.MSELoss()

    def forward(self, x, y=None):
        pred = self.sigmoid(self.linear(x))
        out = {"pred": pred}
        if y is not None:
            out["loss"] = self.mse(pred, y)
        return out

    def _to_params(self):
        return {}

    def training_step(self, batch, batch_idx):
        outputs = self.forward(**batch)
        return outputs["loss"]

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self.forward(**batch)
        return outputs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# TODO: remove replicated code
@Step.register("generate_data")
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
