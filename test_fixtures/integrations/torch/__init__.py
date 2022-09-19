import torch.nn as nn

from tango.integrations.torch import Model


@Model.register("basic_regression")
class BasicRegression(Model):
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
