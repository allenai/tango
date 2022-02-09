import torch
import torch.nn as nn

from tango import Step
from tango.common import DatasetDict
from tango.integrations.torch import Model
from tango.integrations.torch.util import set_seed_all


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.linear(x))


@Model.register("simple_regression_model")
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


@Step.register("simple_regression_data")
class SimpleRegressionDataStep(Step):
    DETERMINISTIC = True
    CACHEABLE = False

    def run(self, seed: int = 317) -> DatasetDict:  # type: ignore
        set_seed_all(seed)

        def get_data(n: int):
            return [{"x": torch.randn(4), "y": torch.randn(1)} for _ in range(n)]

        dataset_dict = DatasetDict(splits={"train": get_data(32), "dev": get_data(16)})
        return dataset_dict
