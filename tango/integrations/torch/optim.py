import torch

from tango.common.registrable import Registrable


class Optimizer(torch.optim.Optimizer, Registrable):
    """
    A :class:`~tango.common.Registrable` version of a PyTorch
    :class:`torch.optim.Optimizer`.

    All `built-in PyTorch optimizers
    <https://pytorch.org/docs/stable/optim.html#algorithms>`_
    are registered according to their class name (e.g. "torch::Adam").

    .. tip::

        You can see a list of all available optimizers by running

        .. testcode::

            from tango.integrations.torch import Optimizer
            for name in sorted(Optimizer.list_available()):
                print(name)

        .. testoutput::
            :options: +ELLIPSIS

            torch::ASGD
            torch::Adadelta
            torch::Adagrad
            torch::Adam
            torch::AdamW
            ...

    """


class LRScheduler(torch.optim.lr_scheduler._LRScheduler, Registrable):
    """
    A :class:`~tango.common.Registrable` version of a PyTorch learning
    rate scheduler.

    All `built-in PyTorch learning rate schedulers
    <https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate>`_
    are registered according to their class name (e.g. "torch::StepLR").

    .. tip::

        You can see a list of all available schedulers by running

        .. testcode::

            from tango.integrations.torch import LRScheduler
            for name in sorted(LRScheduler.list_available()):
                print(name)

        .. testoutput::
            :options: +ELLIPSIS

            torch::ChainedScheduler
            torch::ConstantLR
            torch::CosineAnnealingLR
            ...
    """


# Register all optimizers.
for name, cls in torch.optim.__dict__.items():
    if (
        isinstance(cls, type)
        and issubclass(cls, torch.optim.Optimizer)
        and not cls == torch.optim.Optimizer
    ):
        Optimizer.register("torch::" + name)(cls)

# Register all learning rate schedulers.
for name, cls in torch.optim.lr_scheduler.__dict__.items():
    if (
        isinstance(cls, type)
        and issubclass(cls, torch.optim.lr_scheduler._LRScheduler)
        and not cls == torch.optim.lr_scheduler._LRScheduler
    ):
        LRScheduler.register("torch::" + name)(cls)
