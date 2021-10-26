import typing as t
import torch
import pytorch_lightning as pl

from tango.common.dataset_dict import DatasetDict
from tango.common.lazy import Lazy
from tango.common.registrable import Registrable
from tango.format import Format
from tango.step import Step

from tango.integrations.torch.data import DataLoader
from tango.integrations.torch.format import TorchFormat

from .model import LightningModule
from .loggers import LightningLogger
from .callbacks import LightningCallback
from .profilers import LightningProfiler
from .accelerators import LightningAccelerator


class LightningTrainer(pl.Trainer, Registrable):
    """
    This is simply a :class:`~tango.common.registrable.Registrable`
    :class:`pytorch_lightning.Trainer`.
    """

    def _to_params(self):
        return {}


LightningTrainer.register("default")(LightningTrainer)


@Step.register("pytorch_lightning::train")
class LightningTrainStep(Step):
    """
    A basic PyTorch Lightning Trainer.

    .. tip::

        Registered as a :class:`~tango.step.Step` under the name "pytorch_lightning::train".
    """

    DETERMINISTIC: bool = True
    CACHEABLE = True
    FORMAT: Format = TorchFormat()

    def run(  # type: ignore[override]
        self,
        trainer: Lazy[LightningTrainer],
        model: LightningModule,
        dataset_dict: DatasetDict,
        train_dataloader: Lazy[DataLoader],
        train_split: str = "train",
        *,
        validation_dataloader: Lazy[DataLoader] = None,
        validation_split: str = "validation",
        loggers: t.Optional[t.List[Lazy[LightningLogger]]] = None,
        callbacks: t.Optional[t.List[Lazy[LightningCallback]]] = None,
        profilers: t.Optional[t.List[Lazy[LightningProfiler]]] = None,
        accelerator: t.Optional[Lazy[LightningAccelerator]] = None,
    ) -> torch.nn.Module:  # LightningModule:

        """
        Run a basic training loop to train the ``model``.

        Parameters
        ----------

        trainer : :class:`LightningTrainer`
            The lightning trainer object.
        model : :class:`LightningModule`
            The lightning module to train.
        dataset_dict : :class:`~tango.common.dataset_dict.DatasetDict`
            The train and optional validation data.
        train_dataloader : :class:`DataLoader`
            The data loader that generates training batches. The batches should be :class:`dict`
            objects.
        train_split : :class:`str`, optional
            The name of the data split used for training in the ``dataset_dict``.
            Default is "train".
        validation_split : :class:`str`, optional
            Optional name of the validation split in the ``dataset_dict``. Default is ``None``,
            which means no validation.
        validation_dataloader : :class:`DataLoader`, optional
            An optional data loader for generating validation batches. The batches should be
            :class:`dict` objects. If not specified, but ``validation_split`` is given,
            the validation ``DataLoader`` will be constructed from the same parameters
            as the train ``DataLoader``.
        loggers: ``List[LightningLogger]``
            A list of :class:`LightningLogger`.
        callbacks: ``List[LightningCallback]``
            A list of :class:`LightningCallback`.
        profilers: ``List[LightningProfiler]``
            A list of :class:`LightningProfiler`.
        accelerator: :class:`LightningAccelerator`.
            :class:`LightningAccelerator` object.

        Returns
        -------
        :class:`LightningModule`
            The trained model on CPU with the weights from the best checkpoint loaded.

        """

        # TODO: should we construct the model lazily too?
        # Maybe with optimizer options?

        loggers: t.List[LightningLogger] = [
            logger.construct(save_dir=self.work_dir) for logger in (loggers or [])
        ]

        callbacks: t.List[LightningCallback] = [
            callback.construct() for callback in (callbacks or [])
        ]

        profilers: t.List[LightningProfiler] = [
            profiler.construct(dirpath=self.work_dir) for profiler in (profilers or [])
        ]

        if accelerator:
            accelerator = accelerator.construct()

        trainer: LightningTrainer = trainer.construct(
            logger=loggers, callbacks=callbacks, profilers=profilers, accelerator=accelerator
        )

        checkpoint_callback: pl.callbacks.model_checkpoint.ModelCheckpoint

        for callback in trainer.callbacks:
            if isinstance(callback, pl.callbacks.model_checkpoint.ModelCheckpoint):
                callback.dirpath = self.work_dir
                checkpoint_callback = callback

        # Construct data loaders.
        if validation_split is not None:
            if validation_dataloader is not None:
                validation_dataloader = validation_dataloader.construct(
                    dataset=dataset_dict[validation_split]
                )
            else:
                validation_dataloader = train_dataloader.construct(
                    dataset=dataset_dict[validation_split]
                )
        else:
            validation_dataloader = None
        validation_dataloader: t.Optional[DataLoader] = t.cast(
            t.Optional[DataLoader], validation_dataloader
        )

        try:
            train_dataset = dataset_dict[train_split]
        except KeyError:
            raise KeyError(f"'{train_split}', available keys are {list(dataset_dict.keys())}")

        train_dataloader: DataLoader = train_dataloader.construct(dataset=train_dataset)

        trainer.fit(model, train_dataloader, validation_dataloader)

        best_model = torch.load(checkpoint_callback.best_model_path)

        return best_model
