import pytorch_lightning as pl

from tango.common.registrable import Registrable


class LightningCallback(pl.Callback, Registrable):
    """
    This is simply a :class:`~tango.common.registrable.Registrable`
    version of the PyTorch Lightning :class:`~pytorch_lightning.callbacks.base.Callback`.
    """


# Register all callbacks.
for name, cls in pl.callbacks.__dict__.items():
    if isinstance(cls, type) and issubclass(cls, pl.Callback) and not cls == pl.Callback:
        LightningCallback.register("pytorch_lightning::" + name)(cls)
