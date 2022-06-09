import optax

from inspect import isfunction

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

            from tango.integrations.optax import Optimizer
            for name in sorted(Optimizer.list_available()):
                print(name)

        .. testoutput::
            :options: +ELLIPSIS

            optax::adafactor
            optax::adagrad
            optax::adam
            optax::adamw
            optax::dpsgd
            ...
    """

    def __init__(self, optimizer):
        self.optimizer = optimizer

    def __call__(self, **kwargs):
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

            from tango.integrations.optax import LRScheduler
            for name in sorted(LRScheduler.list_available()):
                print(name)

        .. testoutput::
            :options: +ELLIPSIS

            optax::constant_schedule
            optax::cosine_decay_schedule
            optax::cosine_onecycle_schedule
            ...
    """

    def __init__(self, scheduler):
        self.scheduler = scheduler

    def __call__(self, **kwargs):
        return self.scheduler(**kwargs)


def optimizer_factory(optim_method):
    return Optimizer(optim_method)


def scheduler_factory(scheduler_method):
    return LRScheduler(scheduler_method)


# Register all optimizers.
for name, cls in optax._src.alias.__dict__.items():
    if (
            isfunction(cls)
            and not name.startswith("_")
            and cls.__annotations__
    ):
        factory_func = optimizer_factory(cls)
        Optimizer.register("optax::" + name)(factory_func)

# Register all learning rate schedulers.
for name, cls in optax._src.schedule.__dict__.items():
    if (
            isfunction(cls)
            and not name.startswith("_")
            and cls.__annotations__
    ):
        factory_func = scheduler_factory(cls)
        LRScheduler.register("optax::" + name)(factory_func)

# TODO: Handle inject_hyperparams in training_engine.py
