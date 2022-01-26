from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import torch
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.wrap import enable_wrap, wrap

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

    def enable_wrap(self):
        """
        A convenience wrapper around FairScale's ``enable_wrap()`` context manager.
        """
        return enable_wrap(wrapper_cls=FSDP, **self.as_kwargs())

    def wrap(self, module: torch.nn.Module):
        """
        A convenience wrapper around FairScale's ``wrap()`` function. Should be used within
        the context of :meth:`enable_wrap()`.
        """
        return wrap(module)
