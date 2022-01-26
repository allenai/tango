from typing import TYPE_CHECKING

import torch

from tango.common.registrable import Registrable

if TYPE_CHECKING:
    from collections import OrderedDict


class Model(torch.nn.Module, Registrable):
    """
    This is simply a :class:`~tango.common.Registrable` mixin class that inherits from
    :class:`torch.nn.Module`.
    Its :meth:`~torch.nn.Module.forward()` method should return a :class:`dict` that
    includes the ``loss`` during training and any tracked metrics during validation.
    """

    def load_final_state_dict(self, state_dict: "OrderedDict[str, torch.Tensor]"):
        """
        You can override this method to customize how the state dict from the best
        training checkpoint is loaded after training completes.

        By default it just calls :meth:`torch.nn.Module.load_state_dict()`, but it can
        be useful to override this when you need to ignore errors due to missing or unexpected
        keys that :meth:`~torch.nn.load_state_dict()` might throw.
        """
        self.load_state_dict(state_dict)
