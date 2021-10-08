import torch

from tango.common.registrable import Registrable


class Model(torch.nn.Module, Registrable):
    """
    This is simply a :class:`~tango.common.registrable.Registrable` :class:`torch.nn.Module`
    that returns a :class:`dict` from its :meth:`~torch.nn.Module.forward()` method which
    should include the ``loss`` during training and validation.
    """
