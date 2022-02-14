import torch

from tango.common.registrable import Registrable


class Model(torch.nn.Module, Registrable):
    """
    This is a :class:`~tango.common.Registrable` mixin class that inherits from
    :class:`torch.nn.Module`.
    Its :meth:`~torch.nn.Module.forward()` method should return a :class:`dict` that
    includes the ``loss`` during training and any tracked metrics during validation.
    """
