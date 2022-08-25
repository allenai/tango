import pytorch_lightning as pl
import pytorch_lightning.loggers

from tango.common.registrable import Registrable


class LightningLogger(pl.loggers.Logger, Registrable):
    """
    This is simply a :class:`~tango.common.registrable.Registrable`
    version of the PyTorch Lightning :class:`~pytorch_lightning.loggers.base.LightningLoggerBase`.
    """


# Register all loggers.
for name, cls in pl.loggers.__dict__.items():
    if (
        isinstance(cls, type)
        and issubclass(cls, pl.loggers.Logger)
        and not cls == pl.loggers.Logger
    ):
        LightningLogger.register("pytorch_lightning::" + name)(cls)
