import pytorch_lightning as pl

from tango.common.registrable import Registrable


class LightningModule(pl.LightningModule, Registrable):
    """
    This is simply a :class:`~tango.common.registrable.Registrable`
    version of the PyTorch Lightning :class:`~pytorch_lightning.core.lightning.LightningModule`.
    It includes the following methods:

    * :meth:`~pytorch_lightning.core.lightning.LightningModule.forward()`
    * :meth:`~pytorch_lightning.core.lightning.LightningModule.training_step()`
    * :meth:`~pytorch_lightning.core.lightning.LightningModule.validation_step()`
    * :meth:`~pytorch_lightning.core.lightning.LightningModule.test_step()`
    * :meth:`~pytorch_lightning.core.lightning.LightningModule.configure_optimizers()`
    """
