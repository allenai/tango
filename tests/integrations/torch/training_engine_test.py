import pytest
import torch
import torch.nn as nn

from tango.common.testing import TangoTestCase
from tango.integrations.torch.model import Model
from tango.integrations.torch.training_engine import TorchTrainingEngine


@Model.register("dummy_model")
class DummyModel(Model):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x, y=None):
        return self.linear(x)


@pytest.mark.parametrize(
    "amp",
    (
        pytest.param(
            True,
            id="amp=True",
            marks=[
                pytest.mark.gpu,
                pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Requires CUDA devices"),
            ],
        ),
        pytest.param(False, id="amp=False"),
    ),
)
class TestTorchTrainingEngine(TangoTestCase):
    def test_grad_scaler(self, amp: bool):
        training_engine = TorchTrainingEngine.from_params(
            {
                "train_config": {"step_id": "001", "work_dir": self.TEST_DIR},
                "model": {
                    "type": "dummy_model",
                },
                "optimizer": {"type": "torch::Adam"},
                "amp": amp,
            }
        )

        if amp:
            state_dict = {"training_steps": None}
            training_engine.save_checkpoint(self.TEST_DIR, state_dict)
            saved_grad_scaler = training_engine.grad_scaler
            training_engine.load_checkpoint(self.TEST_DIR)

            assert (self.TEST_DIR / "worker0_grad_scaler.pt").is_file()
            assert training_engine.grad_scaler == saved_grad_scaler
