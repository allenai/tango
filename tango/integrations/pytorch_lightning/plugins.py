import pytorch_lightning as pl

from tango.common.registrable import Registrable


class LightningPlugin(pl.plugins.Plugin, Registrable):
    """
    This is simply a :class:`~tango.common.registrable.Registrable`
    version of the PyTorch Lightning :class:`~pytorch_lightning.plugins.Plugin` class.
    """


# Register all callbacks.
for name, cls in pl.plugins.__dict__.items():
    if (
        isinstance(cls, type)
        and issubclass(cls, pl.plugins.Plugin)
        and not cls == pl.plugins.Plugin
    ):
        LightningPlugin.register("pytorch_lightning::" + name)(cls)
