import torch
from transformers import optimization as transformers_optim

from tango.integrations.torch.optim import LRScheduler, Optimizer

# Register all transformers optimizers.
for name, cls in transformers_optim.__dict__.items():
    if (
        isinstance(cls, type)
        and issubclass(cls, torch.optim.Optimizer)
        and not cls == torch.optim.Optimizer
    ):
        Optimizer.register("transformers::" + name)(cls)


# Register all transformers scheduler factory functions.
for scheduler_type, scheduler_func in transformers_optim.TYPE_TO_SCHEDULER_FUNCTION.items():
    name = scheduler_type.value
    LRScheduler.register("transformers::" + name)(scheduler_func)  # type: ignore
