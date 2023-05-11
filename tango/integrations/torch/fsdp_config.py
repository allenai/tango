from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from tango.common import FromParams


@dataclass
class FSDPConfig(FromParams):
    """
    Defines all of the configurable options for Torch's :class:`~torch.distributed.fsdp.FullyShardedDataParallel`.

    """  # noqa: E501

    move_params_to_cpu: bool = False
    """
    See the docstring for :class:`~torch.distributed.fsdp.FullyShardedDataParallel`.
    """

    move_grads_to_cpu: Optional[bool] = None
    """
    See the docstring for :class:`~torch.distributed.fsdp.FullyShardedDataParallel`.

    .. seealso::
        :data:`move_params_to_cpu`

    .. warning::
        At the moment we recommend that you don't mess with this parameter, or only explicitly
        set it to the same value as :data:`move_params_to_cpu`. If you leave it as ``None``
        (the default), it will automatically be set to match :data:`move_params_to_cpu`.

        Currently training seems to crash if you set this ``False`` while :data:`move_params_to_cpu` is ``True``.
        We're tracking `fairscale#918 <https://github.com/facebookresearch/fairscale/issues/918>`_,
        which may be related.
    """

    mixed_precision: bool = False
    """
    See the docstring for :class:`~torch.distributed.fsdp.FullyShardedDataParallel`.

    .. important::
        We recommend setting this to the same value as the ``amp`` parameter in
        :class:`FSDPTrainingEngine`.

        Based on our experiments, if you're training with AMP enabled (``amp=True``)
        you might see a small additional speedup in training time along with a small
        additional decrease in GPU memory utilization without any performance penalty
        (with respect to convergence) by setting this to ``True``.
        But if you're *not* training with AMP, setting this ``True`` could impact the
        model's ability to converge.

    """

    def as_kwargs(self) -> Dict[str, Any]:
        """
        Convert to the appropriate ``kwargs`` for :class:`~torch.distributed.fsdp.FullyShardedDataParallel`.
        """
        return asdict(self)

    def wrap(self, module: torch.nn.Module):
        """
        A convenience method for wrapping a module in :class:`~torch.distributed.fsdp.FullyShardedDataParallel`
        with all the options defined in this class.

        .. seealso::
            Internally this is what :func:`with_wrapped_modules()` calls.

        """
        return FSDP(module, **self.as_kwargs())
