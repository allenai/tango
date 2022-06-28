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
    "FlaxTrainStep",
    "FlaxTrainWrapper",
    "TrainConfig",
    "LossFunction",
    "FlaxEvalStep",
    "FlaxEvalWrapper",
]

from .data import DataLoader, FlaxDataLoader
from .eval import FlaxEvalStep, FlaxEvalWrapper
from .loss import LossFunction
from .model import Model
from .optim import LRScheduler, Optimizer
from .train import FlaxTrainStep
from .train_config import TrainConfig
from .train_state import FlaxTrainWrapper
