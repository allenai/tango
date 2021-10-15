"""
Components for Tango integration with `Weights & Biases <https://wandb.ai/>`_.

.. important::
    To use this integration you should install ``tango`` with the "wandb" extra
    (e.g. ``pip install tango[wandb]``) or just install the ``wandb`` library after the fact
    (e.g. ``pip install wandb``).

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
