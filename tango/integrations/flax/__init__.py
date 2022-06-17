from tango.common.exceptions import IntegrationMissingError

try:
    import flax
except ModuleNotFoundError:
    raise IntegrationMissingError("flax")

__all__ = [
    "DataLoader",
    "FlaxDataLoader",
    "LRScheduler",
    "Model",
    "Optimizer",
    "TrainState",
    "FlaxTrainStep",
    "LossFunction",
]

from .data import DataLoader, FlaxDataLoader
from .loss import LossFunction
from .model import Model
from .optim import LRScheduler, Optimizer
from .train import FlaxTrainStep
from .train_state import TrainState
