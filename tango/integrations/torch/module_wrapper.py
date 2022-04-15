from dataclasses import dataclass
from typing import Optional, Set

from tango.common import Registrable
from tango.integrations.torch import Model

from .fsdp_config import TorchFSDPConfig


@dataclass
class ModuleWrapper(Registrable):
    # TODO: improve docstrings.

    modules_to_wrap: Set[str]
    """
    The names of submodule to wrap. Can be regular expressions.
    """

    fsdp_config: Optional[TorchFSDPConfig] = None
    """
    The ``FullyShardedDataParallel`` configuration to use when wrapping the modules.
    If not specified, the modules will NOT be wrapped with FSDP.
    """

    activation_checkpointing: bool = False
    """
    Whether to wrap the module in activation checkpointer.
    """

    def with_wrapped_modules(self, model: Model) -> Model:
        # TODO: default implementation.
        raise NotImplementedError()
