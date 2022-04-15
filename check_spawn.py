import torch.multiprocessing as mp
import torch.nn as nn

from tango.integrations.torch import Model


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


model_fair = Model.from_params(
    {
        "type": "fairscale::with_wrapped_modules",
        "model": {
            "type": "simple_regression_model",
        },
        "modules_to_wrap": [r"blocks\.[0-9]+", "regression_head"],
        "activation_checkpointing": True,
    }
)

model_sim = Model.from_params({"type": "simple_regression_model"})


def func(worker_id, model):
    print(worker_id)
    print(model)


def main(model):
    print("Before")
    print(model)
    print("End")
    mp.spawn(func, args=(model,), nprocs=2)


if __name__ == "__main__":
    main(model_fair)
