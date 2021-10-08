import torch

from tango.common.registrable import Registrable


class Optimizer(torch.optim.Optimizer, Registrable):
    """
    A :class:`~tango.common.registrable.Registrable` version of a PyTorch
    :class:`torch.optim.Optimizer`.

    All `built-in PyTorch optimizers
    <https://pytorch.org/docs/stable/optim.html#algorithms>`_
    are registered under their corresponding class name (e.g. "Adam").
    """


class LRScheduler(torch.optim.lr_scheduler._LRScheduler, Registrable):
    """
    A :class:`~tango.common.registrable.Registrable` version of a PyTorch learning
    rate scheduler.

    All `built-in PyTorch learning rate schedulers
    <https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate>`_
    are registered under their corresponding class name (e.g. "StepLR").
    """


# Register all optimizers.
for name, cls in torch.optim.__dict__.items():
    if (
        isinstance(cls, type)
        and issubclass(cls, torch.optim.Optimizer)
        and not cls == torch.optim.Optimizer
    ):
        Optimizer.register(name)(cls)

# Register all learning rate schedulers.
for name, cls in torch.optim.lr_scheduler.__dict__.items():
    if (
        isinstance(cls, type)
        and issubclass(cls, torch.optim.lr_scheduler._LRScheduler)
        and not cls == torch.optim.lr_scheduler._LRScheduler
    ):
        LRScheduler.register(name)(cls)
