import pytorch_lightning as pl

from tango.common.registrable import Registrable


class LightningAccelerator(pl.accelerators.Accelerator, Registrable):
    """
    This is simply a :class:`~tango.common.registrable.Registrable`
    version of the PyTorch Lightning :class:`~pytorch_lightning.accelerators.Accelerator`.
    """


# Register all accelerators.
for name, cls in pl.accelerators.__dict__.items():
    if (
        isinstance(cls, type)
        and issubclass(cls, pl.accelerators.Accelerator)
        and not cls == pl.accelerators.Accelerator
    ):
        LightningAccelerator.register("pytorch_lightning::" + name)(cls)
