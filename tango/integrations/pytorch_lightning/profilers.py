import pytorch_lightning as pl

from tango.common.registrable import Registrable


class LightningProfiler(pl.profiler.base.BaseProfiler, Registrable):
    """
    This is simply a :class:`~tango.common.registrable.Registrable`
    version of the PyTorch Lightning :class:`~pytorch_lightning.profiler.base.BaseProfiler`.
    """


# Register all profilers.
for name, cls in pl.profiler.__dict__.items():
    if (
        isinstance(cls, type)
        and issubclass(cls, pl.profiler.base.BaseProfiler)
        and not cls == pl.profiler.base.BaseProfiler
    ):
        LightningProfiler.register("pytorch_lightning::" + name)(cls)
