import re
from typing import Optional, Set

import torch
import torch.nn as nn
from fairscale.nn.checkpoint import checkpoint_wrapper

from tango.integrations.torch import Model

from .fsdp_config import FSDPConfig


@Model.register("fairscale::with_wrapped_modules")  # type: ignore[arg-type]
def with_wrapped_modules(
    model: Model,
    modules_to_wrap: Set[str],
    fsdp_config: Optional[FSDPConfig] = None,
    activation_checkpointing: bool = False,
) -> Model:
    """
    A :class:`~tango.integrations.torch.Model` wrapper that can be used to easily wrap
    inner modules of a model with FairScale's :class:`~fairscale.nn.FullyShardedDataParallel` wrapper
    and/or :class:`~fairscale.nn.checkpoint.checkpoint_wrapper`.

    .. tip::
        Registered as a :class:`~tango.integrations.torch.Model` constructor under the name
        "fairscale::with_wrapped_modules".

    Parameters
    ----------

    model : :class:`~tango.integrations.torch.Model`
        The model to wrap.
    modules_to_wrap : :class:`set`
        The names of submodule to wrap. Can be regular expressions.
    fsdp_config : :class:`~FSDPConfig`, optional
        The ``FullyShardedDataParallel`` configuration to use when wrapping the modules.
    activation_checkpointing : :class:`bool`, optional
        Whether to wrap the modules with FairScale's
        :class:`~fairscale.nn.checkpoint.checkpoint_wrapper`.

    """

    def wrap_module(
        module: nn.Module,
    ) -> nn.Module:
        if activation_checkpointing:
            module = checkpoint_wrapper(module, offload_to_cpu=True)
        if fsdp_config is not None and torch.distributed.is_initialized():
            module = fsdp_config.wrap(module)
        return module

    all_module_names: Set[str] = set([name for name, _ in model.named_modules() if name])
    actual_modules_to_wrap: Set[str] = set()
    unmatched_patterns: Set[str] = modules_to_wrap.copy()
    for module_name in all_module_names:
        for pattern in modules_to_wrap:
            if re.fullmatch(pattern, module_name):
                actual_modules_to_wrap.add(module_name)
                unmatched_patterns.remove(pattern)

    if unmatched_patterns:
        raise ValueError(
            f"Some patterns in 'modules_to_wrap' did not match actual module names ({unmatched_patterns})"
        )

    for module_name in actual_modules_to_wrap:
        if "." in module_name:
            parent_module_name, module_name = module_name.split(".", 1)
            parent_module = model.get_submodule(parent_module_name)
        else:
            parent_module = model
        module = parent_module.get_submodule(module_name)
        if isinstance(module, (nn.ModuleList, nn.Sequential)):
            for i in range(len(module)):
                module[i] = wrap_module(module[i])
        else:
            module = wrap_module(module)

    return model
