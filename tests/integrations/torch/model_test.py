import pytest
from torch import nn

from tango.common.testing import TangoTestCase
from tango.integrations.torch import Model
from tango.step import Step


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.linear(x))


@Model.register("simple_regression_model", exist_ok=True)
class SimpleRegressionModel(Model):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(*[FeedForward() for _ in range(3)])
        self.regression_head = nn.Linear(4, 1)
        self.loss_fcn = nn.MSELoss()

    def forward(self, x, y):
        output = self.blocks(x)
        output = self.regression_head(output)
        loss = self.loss_fcn(output, y)
        return {"loss": loss}


@Step.register("step-that-takes-model-as-input")
class StepThatTakesModelAsInput(Step):
    def run(self, model: Model) -> Model:  # type: ignore
        return model


class TestModelAsStepInput(TangoTestCase):
    def test_step_that_takes_model_as_input(self):
        config = {
            "steps": {
                "model": {
                    "type": "step-that-takes-model-as-input",
                    "model": {"type": "simple_regression_model"},
                }
            }
        }

        with pytest.raises(NotImplementedError):
            self.run(config)
