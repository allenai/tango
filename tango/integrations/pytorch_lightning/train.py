import logging
from pathlib import Path
from typing import List, Optional, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.plugins import PLUGIN_INPUT

from tango.common.dataset_dict import DatasetDict
from tango.common.lazy import Lazy
from tango.common.registrable import Registrable
from tango.format import Format
from tango.integrations.torch.data import DataLoader
from tango.integrations.torch.format import TorchFormat
from tango.step import Step

from .accelerators import LightningAccelerator
from .callbacks import LightningCallback
from .data import LightningDataModule
from .loggers import LightningLogger
from .model import LightningModule
from .plugins import LightningTrainingTypePlugin
from .profilers import LightningProfiler

logger = logging.getLogger(__name__)


class LightningTrainer(pl.Trainer, Registrable):  # type: ignore
    """
    This is simply a :class:`~tango.common.Registrable` version of
    the PyTorch Lightning :class:`~pytorch_lightning.trainer.trainer.Trainer`.
    """

    def __init__(
        self,
        work_dir: Path,
        logger: Optional[Union[List[Lazy[LightningLogger]], Lazy[LightningLogger]]] = None,
        callbacks: Union[List[LightningCallback], LightningCallback, None] = None,
        profiler: Optional[Union[str, Lazy[LightningProfiler]]] = None,
        accelerator: Optional[Union[str, LightningAccelerator]] = None,
        strategy: Optional[Union[str, LightningTrainingTypePlugin]] = None,
        plugins: Optional[Union[PLUGIN_INPUT, List[PLUGIN_INPUT]]] = None,
        **kwargs,
    ):
        loggers: List[LightningLogger] = (
            []
            if not logger
            else [
                logger_.construct(save_dir=work_dir)
                for logger_ in (logger if isinstance(logger, list) else [logger])
            ]
        )

        profiler: Optional[Union[str, LightningProfiler]] = (
            profiler.construct(dirpath=work_dir) if isinstance(profiler, Lazy) else profiler
        )

        super().__init__(
            logger=loggers,
            callbacks=callbacks,  # type: ignore  # (type is correct but mypy can't figure it out)
            profiler=profiler,
            accelerator=accelerator,
            strategy=strategy,
            plugins=plugins,
            **kwargs,
        )


LightningTrainer.register("default")(LightningTrainer)


@Step.register("pytorch_lightning::train")
class LightningTrainStep(Step):
    """
    A step for training a model using PyTorch Lightning.

    .. tip::

        Registered as a :class:`~tango.step.Step` under the name "pytorch_lightning::train".
    """

    DETERMINISTIC: bool = True
    CACHEABLE = True
    FORMAT: Format = TorchFormat()
    METADATA = {"artifact_kind": "model"}

    def run(  # type: ignore[override]
        self,
        trainer: Lazy[LightningTrainer],
        model: Union[Lazy[LightningModule], LightningModule],  # Lazy has to come first
        *,
        dataset_dict: Optional[DatasetDict] = None,
        train_dataloader: Optional[Lazy[DataLoader]] = None,
        train_split: str = "train",
        validation_dataloader: Optional[Lazy[DataLoader]] = None,
        validation_split: str = "validation",
        datamodule: Optional[LightningDataModule] = None,
    ) -> torch.nn.Module:

        """
        Run a basic training loop to train the ``model``.

        :param trainer:
            The lightning trainer object.
        :param model:
            The lightning module to train.
        :param dataset_dict:
            The train and optional validation data. This is ignored if the `datamodule` argument
            is provided.
        :param train_dataloader:
            The data loader that generates training batches. The batches should be :class:`dict`
            objects. This is ignored if the `datamodule` argument is provided.
        :param train_split:
            The name of the data split used for training in the ``dataset_dict``.
            Default is "train". This is ignored if the `datamodule` argument is provided.
        :param validation_split:
            Optional name of the validation split in the ``dataset_dict``. Default is ``None``,
            which means no validation. This is ignored if the `datamodule` argument is provided.
        :param validation_dataloader:
            An optional data loader for generating validation batches. The batches should be
            :class:`dict` objects. If not specified, but ``validation_split`` is given,
            the validation ``DataLoader`` will be constructed from the same parameters
            as the train ``DataLoader``. This is ignored if the `datamodule` argument
            is provided.
        :param datamodule:
            If a :class:`LightningDataModule` object is given, the other data loading arguments
            are ignored.

        :returns:
            The trained model on CPU with the weights from the best checkpoint loaded.

        """
        trainer: LightningTrainer = trainer.construct(work_dir=self.work_dir)
        model_: LightningModule = model.construct() if isinstance(model, Lazy) else model

        # Find the checkpoint callback and make sure it uses the right directory.
        checkpoint_callback: pl.callbacks.model_checkpoint.ModelCheckpoint
        for callback in trainer.callbacks:
            if isinstance(callback, pl.callbacks.model_checkpoint.ModelCheckpoint):
                callback.dirpath = self.work_dir
                checkpoint_callback = callback

        if datamodule:
            logger.info(
                "A datamodule object has been given. Other data loading arguments "
                "will be ignored!"
            )
            trainer.fit(model_, datamodule=datamodule)
        elif dataset_dict and train_dataloader:
            # Construct data loaders.
            validation_dataloader_: Optional[DataLoader] = None
            if validation_split is not None:
                if validation_dataloader is not None:
                    validation_dataloader_ = validation_dataloader.construct(
                        dataset=dataset_dict[validation_split]
                    )
                else:
                    validation_dataloader_ = train_dataloader.construct(
                        dataset=dataset_dict[validation_split]
                    )
            validation_dataloader: Optional[DataLoader] = validation_dataloader_  # type: ignore
            try:
                train_dataset = dataset_dict[train_split]
            except KeyError:
                raise KeyError(f"'{train_split}', available keys are {list(dataset_dict.keys())}")
            train_dataloader: DataLoader = train_dataloader.construct(dataset=train_dataset)  # type: ignore
            trainer.fit(model_, train_dataloader, validation_dataloader)  # type: ignore
        else:
            raise AssertionError(
                "You need to provide either the `datamodule` argument, "
                "or `dataset_dict` and `train_dataloader` (and other data loading arguments)."
            )

        best_model = torch.load(checkpoint_callback.best_model_path)
        return best_model
