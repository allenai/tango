from inspect import isfunction
from typing import Callable, Type

import optax

from tango.common.registrable import Registrable


class LossFunction(Registrable):
    """
    register different loss functions for use in train and eval.
    """

    def __init__(self, loss: Callable) -> None:
        self.loss = loss

    def __call__(self, **kwargs):
        return self.loss(**kwargs)


def loss_factory(loss: Callable) -> Type[Callable]:
    def factory_func():
        return LossFunction(loss)

    return factory_func()


for name, cls in optax._src.loss.__dict__.items():
    if isfunction(cls) and not name.startswith("_") and cls.__annotations__:
        factory_func = loss_factory(cls)
        LossFunction.register("optax::" + name)(factory_func)
