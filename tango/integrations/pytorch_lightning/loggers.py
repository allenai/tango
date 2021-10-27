import pytorch_lightning as pl

from tango.common.registrable import Registrable


class LightningLogger(pl.loggers.LightningLoggerBase, Registrable):
    """
    This is simply a :class:`~tango.common.registrable.Registrable`
    version of the PyTorch Lightning :class:`~pytorch_lightning.loggers.base.LightningLoggerBase`.
    """


# Register all loggers.
for name, cls in pl.loggers.__dict__.items():
    if (
        isinstance(cls, type)
        and issubclass(cls, pl.loggers.LightningLoggerBase)
        and not cls == pl.loggers.LightningLoggerBase
    ):
        LightningLogger.register("pytorch_lightning::" + name)(cls)
