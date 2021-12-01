from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn

from tango.common.dataset_dict import DatasetDict
from tango.integrations.pytorch_lightning import LightningDataModule, LightningModule
from tango.integrations.torch.data import DataLoader
from tango.step import Step


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


@LightningDataModule.register("generate_data_module")
class GenerateDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        eval_batch_size: int = 32,
        shuffle: bool = False,
        eval_shuffle: bool = False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.eval_batch_size = batch_size
        self.shuffle = shuffle
        self.eval_shuffle = eval_shuffle

    def setup(self, stage: Optional[str] = None):
        torch.manual_seed(1)
        self.dataset_dict = DatasetDict(
            {
                "train": [{"x": torch.rand(10), "y": torch.rand(1)} for _ in range(64)],
                "validation": [{"x": torch.rand(10), "y": torch.rand(1)} for _ in range(32)],
            }
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.dataset_dict["train"], batch_size=self.batch_size, shuffle=self.shuffle  # type: ignore
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.dataset_dict["validation"],  # type: ignore
            batch_size=self.eval_batch_size,
            shuffle=self.eval_shuffle,
        )
