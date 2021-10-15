"""
Components for Tango integration with `Weights & Biases <https://wandb.ai/>`_.
"""

__all__ = []

try:
    from .torch_train_callback import WandbTrainCallback

    __all__.append("WandbTrainCallback")
except ModuleNotFoundError as exc:
    if exc.name == "torch":
        pass
    else:
        raise
