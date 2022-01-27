from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import torch
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP

from tango.common import FromParams


@dataclass
class FSDPConfig(FromParams):
    """
    Defines all of the configurable options for FairScale's :class:`~fairscale.nn.FullyShardedDataParallel`.
    """

    reshard_after_forward: bool = True
    """
    See the docstring for :class:`~fairscale.nn.FullyShardedDataParallel`.
    """

    move_params_to_cpu: bool = False
    """
    See the docstring for :class:`~fairscale.nn.FullyShardedDataParallel`.
    """

    move_grads_to_cpu: Optional[bool] = None
    """
    See the docstring for :class:`~fairscale.nn.FullyShardedDataParallel`.
    """

    mixed_precision: bool = False
    """
    See the docstring for :class:`~fairscale.nn.FullyShardedDataParallel`.
    """

    def as_kwargs(self) -> Dict[str, Any]:
        """
        Convert to the appropriate ``kwargs`` for :class:`~fairscale.nn.FullyShardedDataParallel`.
        """
        return asdict(self)

    def wrap(self, module: torch.nn.Module):
        """
        A convenience method for wrapping a module in :class:`~fairscale.nn.FullyShardedDataParallel`
        with all of the options defined in this class.
        """
        return FSDP(module, **self.as_kwargs())
