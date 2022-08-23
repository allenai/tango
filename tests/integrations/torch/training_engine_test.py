import time

import pytest
import torch
import torch.nn as nn
from torch.nn import MSELoss

from tango.common import DatasetDict, Lazy
from tango.common.testing import TangoTestCase
from tango.integrations.torch import (
    DataLoader,
    StopEarly,
    TorchTrainStep,
    TrainCallback,
)
from tango.integrations.torch.model import Model
from tango.integrations.torch.training_engine import TorchTrainingEngine


@Model.register("dummy_model")
class DummyModel(Model):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x, y=None):
        return self.linear(x)


@pytest.mark.gpu
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Requires CUDA devices")
class TestTorchTrainingEngine(TangoTestCase):
    def test_grad_scaler(self):
        training_engine = TorchTrainingEngine.from_params(
            {
                "train_config": {"step_id": "001", "work_dir": self.TEST_DIR},
                "model": {
                    "type": "dummy_model",
                },
                "optimizer": {"type": "torch::Adam"},
                "amp": True,
            }
        )

        state_dict = {"training_steps": None}
        training_engine.save_checkpoint(self.TEST_DIR, state_dict)
        saved_grad_scaler = training_engine.grad_scaler
        training_engine.load_checkpoint(self.TEST_DIR)

        assert (self.TEST_DIR / "worker0_grad_scaler.pt").is_file()
        assert training_engine.grad_scaler == saved_grad_scaler


class WorseningModel(Model):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(7, 1)
        self.loss = MSELoss()
        self.start_time = time.time()

    def forward(self, x, y):
        y_hat = self.linear(x)
        time.sleep(0.01)
        return {"loss": self.loss(y_hat, y) + (time.time() - self.start_time)}


class StopOnStepCallback(TrainCallback):
    def __init__(self, stop_on_step: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stop_on_step = stop_on_step

    def post_val_loop(
        self, step: int, epoch: int, val_metric: float, best_val_metric: float
    ) -> None:
        if step == self.stop_on_step:
            raise StopEarly


def test_with_increasing_loss():
    model = WorseningModel()

    xs = [torch.randn(7) for _ in range(100)]
    train_set = [{"x": x, "y": x + 0.1} for x in xs]
    dataset = DatasetDict(splits={"train": train_set, "validation": train_set}, metadata={})

    step = TorchTrainStep(
        model=model,
        training_engine=Lazy(TorchTrainingEngine, optimizer=Lazy(torch.optim.AdamW, lr=1e-5)),
        dataset_dict=dataset,
        train_dataloader=Lazy(DataLoader),
        train_steps=10,
        validation_steps=10,
        train_split="train",
        validation_split="validation",
        callbacks=[Lazy(StopOnStepCallback, stop_on_step=9)],
    )
    step.result()
