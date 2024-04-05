from inspect import isfunction
from typing import Callable, Type

import optax

from tango.common.registrable import Registrable


class Optimizer(Registrable):
    """
    A :class:`~tango.common.Registrable` version of Optax optimizers.

    All `built-in Optax optimizers
    <https://optax.readthedocs.io/en/latest/api.html#>`_
    are registered according to their class name (e.g. "optax::adam").

    .. tip::

        You can see a list of all available optimizers by running

        .. testcode::

            from tango.integrations.flax import Optimizer
            for name in sorted(Optimizer.list_available()):
                print(name)

        .. testoutput::
            :options: +ELLIPSIS

            optax::adabelief
            optax::adadelta
            optax::adafactor
            optax::adagrad
            optax::adam
            ...

    """

    def __init__(self, optimizer: Callable) -> None:
        self.optimizer = optimizer

    def __call__(self, **kwargs) -> optax.GradientTransformation:
        return self.optimizer(**kwargs)


class LRScheduler(Registrable):
    """
    A :class:`~tango.common.Registrable` version of an Optax learning
    rate scheduler.

    All `built-in Optax learning rate schedulers
    <https://optax.readthedocs.io/en/latest/api.html#schedules>`_
    are registered according to their class name (e.g. "optax::linear_schedule").

    .. tip::

        You can see a list of all available schedulers by running

        .. testcode::

            from tango.integrations.flax import LRScheduler
            for name in sorted(LRScheduler.list_available()):
                print(name)

        .. testoutput::
            :options: +ELLIPSIS

            optax::constant_schedule
            optax::cosine_decay_schedule
            optax::cosine_onecycle_schedule
            optax::exponential_decay
            ...

    """

    def __init__(self, scheduler: Callable) -> None:
        self.scheduler = scheduler

    def __call__(self, **kwargs):
        return self.scheduler(**kwargs)


def optimizer_factory(optim_method: Callable) -> Type[Callable]:
    def factory_func():
        return Optimizer(optim_method)

    return factory_func()


def scheduler_factory(scheduler_method: Callable) -> Type[Callable]:
    def factory_func():
        return LRScheduler(scheduler_method)

    return factory_func()


# Register all optimizers.
for name, cls in optax._src.alias.__dict__.items():
    if isfunction(cls) and not name.startswith("_") and cls.__annotations__:
        factory_func = optimizer_factory(cls)
        Optimizer.register("optax::" + name)(factory_func)

# Register all learning rate schedulers.
for name, cls in optax.schedules.__dict__.items():
    if isfunction(cls) and not name.startswith("_") and cls.__annotations__:
        factory_func = scheduler_factory(cls)
        LRScheduler.register("optax::" + name)(factory_func)

# TODO: Handle inject_hyperparams.
# Refer: https://optax.readthedocs.io/en/latest/api.html?highlight=inject%20hyperparam
