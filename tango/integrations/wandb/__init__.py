"""
.. important::
    To use this integration you should install ``tango`` with the "wandb" extra
    (e.g. ``pip install tango[wandb]``) or just install the ``wandb`` library after the fact
    (e.g. ``pip install wandb``).

Components for Tango integration with `Weights & Biases <https://wandb.ai/>`_.

Overview
--------

The main components provided by this integration are the :class:`WandbWorkspace` and
the :class:`WandbTrainCallback`.

The :class:`WandbWorkspace` is a :class:`~tango.workspace.Workspace` implementation that is
great for collaboration. It tracks Tango runs and steps in the W&B project of your choosing
and uses W&B Artifacts to cache step results in the cloud so that they're accessible anywhere.

And if you're training PyTorch models via the :class:`~tango.integrations.torch.TorchTrainStep`,
you can use the :class:`WandbTrainCallback` to track metrics throughout the run.

"""

__all__ = ["WandbWorkspace", "WandbStepCache"]

from .step_cache import WandbStepCache
from .workspace import WandbWorkspace

try:
    from .torch_train_callback import WandbTrainCallback

    __all__.append("WandbTrainCallback")
except ModuleNotFoundError as exc:
    if exc.name == "torch":
        pass
    else:
        raise
