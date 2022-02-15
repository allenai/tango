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

    .. important::
        This is meant to be used with the :class:`FairScaleTrainingEngine`.

    :param model:
        The model to wrap.
    :param modules_to_wrap:
        The names of submodule to wrap. Can be regular expressions.
    :param fsdp_config:
        The ``FullyShardedDataParallel`` configuration to use when wrapping the modules.
        If not specified, the modules will NOT be wrapped with FSDP.
    :param activation_checkpointing:
        Whether to wrap the modules with FairScale's
        :class:`~fairscale.nn.checkpoint.checkpoint_wrapper`.

    Examples
    --------

    You can use this as a :class:`~tango.integrations.torch.Model` constructor from a config/params
    like this:

    .. testcode::

        import torch.nn as nn
        from tango.integrations.torch import Model


        class FeedForward(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 4)
                self.activation = nn.ReLU()

            def forward(self, x):
                return self.activation(self.linear(x))

        @Model.register("simple_regression_model")
        class SimpleRegressionModel(Model):
            def __init__(self):
                super().__init__()
                self.blocks = nn.Sequential(*[FeedForward() for _ in range(3)])
                self.regression_head = nn.Linear(4, 1)
                self.loss_fcn = nn.MSELoss()

            def forward(self, x, y):
                output = self.blocks(x)
                output = self.regression_head(output)
                loss = self.loss_fcn(output, y)
                return {"loss": loss}


        model = Model.from_params({
            "type": "fairscale::with_wrapped_modules",
            "model": {
                "type": "simple_regression_model",
            },
            "modules_to_wrap": [r"blocks\\.[0-9]+", "regression_head"],
            "activation_checkpointing": True,
        })

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
                if pattern in unmatched_patterns:
                    unmatched_patterns.remove(pattern)

    if unmatched_patterns:
        raise ValueError(
            f"Some patterns in 'modules_to_wrap' did not match actual module names ({unmatched_patterns})"
        )

    for module_name in actual_modules_to_wrap:
        if "." in module_name:
            *parent_parts, module_name = module_name.split(".")
            parent_module = model.get_submodule(".".join(parent_parts))
        else:
            parent_module = model
        module = parent_module.get_submodule(module_name)
        module = wrap_module(module)
        parent_module.add_module(module_name, module)

    return model
