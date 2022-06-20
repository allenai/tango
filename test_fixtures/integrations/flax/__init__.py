from tango.integrations.flax import Model
from flax import linen as nn

from typing import Tuple


@Model.register("classification")
class BasicClassification(Model):
    """
    A simple classification model.
    """
    input_size = 10
    hidden_size = 4

    def setup(self) -> None:
        self.dense1 = nn.Dense(self.input_size)
        self.dense2 = nn.Dense(self.hidden_size)

    @nn.compact
    def __call__(self, x):
        x = self.dense1(x)
        x = nn.relu(x)
        x = self.dense2(x)
        x = nn.softmax(x)
        return x
