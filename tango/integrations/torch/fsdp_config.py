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

    sharding_strategy: Optional[str] = None

    cpu_offload: Optional[str] = None

    backward_prefetch: Optional[str] = None

    sync_module_states: bool = False

    forward_prefetch: bool = False

    limit_all_gathers: bool = False

    use_orig_params: bool = False

    mixed_precision: Optional[str] = None
    """
    See the docstring for :class:`~torch.distributed.fsdp.FullyShardedDataParallel`.

    .. important::
        We recommend setting this to the same value as the ``amp`` parameter in
        :class:`FSDPTrainingEngine`.

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
