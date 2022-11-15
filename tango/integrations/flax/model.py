from flax import linen as nn

from tango.common.registrable import Registrable


class Model(nn.Module, Registrable):
    """
    This is a :class:`~tango.common.Registrable` mixin class that inherits from
    :class:`flax.linen.Module`.
    Its :meth:`~flax.linen.Module.setup()` can be used to register submodules,
    variables, parameters you will need in your model.
    Its :meth:`~flax.linen.Module.__call__()` returns the output of the model
    for a given input.
    """
