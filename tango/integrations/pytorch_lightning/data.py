import pytorch_lightning as pl

from tango.common.registrable import Registrable


class LightningDataModule(pl.LightningDataModule, Registrable):
    """
    This is simply a :class:`~tango.common.registrable.Registrable`
    :class:`pytorch_lightning.LightningDataModule`. It includes the following
    methods:

    * :meth:`~pytorch_lightning.extensions.datamodules.LightningDataModule.prepare_data()`
    * :meth:`~pytorch_lightning.extensions.datamodules.LightningDataModule.setup()`
    * :meth:`~pytorch_lightning.extensions.datamodules.LightningDataModule.train_dataloader()`
    * :meth:`~pytorch_lightning.extensions.datamodules.LightningDataModule.val_dataloader()`
    * :meth:`~pytorch_lightning.extensions.datamodules.LightningDataModule.test_dataloader()`
    """

    def _to_params(self):
        return {}
