import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from tango.common.dataset_dict import DatasetDictBase
from tango.common.registrable import Registrable

from .data import DataLoader
from .exceptions import StopEarly
from .model import Model
from .train_config import TrainConfig
from .training_engine import TrainingEngine


class TrainCallback(Registrable):
    """
    A ``TrainCallback`` is a :class:`~tango.common.Registrable` class
    that can be used within :class:`TorchTrainStep` to customize behavior in the training
    loop.

    .. tip::
        All of the parameters to this base class will be automatically set within
        the training loop, so you shouldn't include them in your config for your callbacks.

    .. tip::
        You can access the model being trained through :attr:`self.model <model>`.

    .. important::
        The ``step`` argument to callback methods is the total/overall number of training steps
        so far, independent of the current epoch.

    .. seealso::
        See :class:`~tango.integrations.wandb.WandbTrainCallback` for an example
        implementation.

    Attributes
    ----------
    train_config : :class:`TrainConfig`
    training_engine : :class:`TrainingEngine`
    optimizer : :class:`Optimizer`
    dataset_dict : :class:`tango.common.DatasetDictBase`
    train_dataloader : :class:`DataLoader`
    validation_dataloader : :class:`DataLoader`, optional
    lr_scheduler : :class:`LRScheduler`, optional

    """

    def __init__(
        self,
        train_config: TrainConfig,
        training_engine: TrainingEngine,
        dataset_dict: DatasetDictBase,
        train_dataloader: DataLoader,
        validation_dataloader: Optional[DataLoader] = None,
    ) -> None:
        self.train_config = train_config
        self.training_engine = training_engine
        self.dataset_dict = dataset_dict
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    def step_id(self) -> str:
        """
        The unique ID of the current :class:`~tango.Step`.
        """
        return self.train_config.step_id

    @property
    def work_dir(self) -> Path:
        """
        The working directory of the current train step.
        """
        return self.train_config.work_dir

    @property
    def is_local_main_process(self) -> bool:
        """
        This is ``True`` if the current worker is the main distributed worker of the current node, or if
        we are not using distributed training.
        """
        return self.train_config.is_local_main_process

    @property
    def model(self) -> Model:
        """
        The :class:`Model` being trained.
        """
        return self.training_engine.model

    def state_dict(self) -> Dict[str, Any]:
        """
        Return any state that needs to be kept after a restart.

        Some callbacks need to maintain state across restarts. This is the callback's opportunity to
        save it's state. It will be restored using :meth:`load_state_dict`.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load the state on a restart.

        Some callbacks need to maintain state across restarts. This is the callback's opportunity to
        restore it's state. It gets saved using :meth:`state_dict`.
        """
        pass

    def pre_train_loop(self) -> None:
        """
        Called right before the first batch is processed, or after a restart.
        """
        pass

    def post_train_loop(self, step: int, epoch: int) -> None:
        """
        Called after the training loop completes.

        This is the last method that is called, so any cleanup can be done in this method.
        """
        pass

    def pre_epoch(self, step: int, epoch: int) -> None:
        """
        Called right before the start of an epoch. Epochs start at 0.
        """
        pass

    def post_epoch(self, step: int, epoch: int) -> None:
        """
        Called after an epoch is completed. Epochs start at 0.
        """
        pass

    def pre_batch(self, step: int, epoch: int, batch: List[Dict[str, Any]]) -> None:
        """
        Called directly before processing a batch.

        .. note::
            A type of ``batch`` is a list because with gradient accumulation there will
            more than one "micro batch" in the batch.

        """
        pass

    def post_batch(self, step: int, epoch: int, batch_loss: float) -> None:
        """
        Called directly after processing a batch, but before unscaling gradients,
        clipping gradients, and taking an optimizer step.

        .. note::
            The ``batch_loss`` here is the loss local to the current worker, not the
            overall (average) batch loss across distributed workers.

            If you need the average loss, use :meth:`log_batch()`.

        """
        pass

    def log_batch(self, step: int, epoch: int, batch_loss: float) -> None:
        """
        Called after the optimizer step. Here ``batch_loss`` is the average loss across
        all distributed workers.

        .. note::
            This callback method is not necessarily called on every step.
            The frequency depends on the value of the ``log_every`` parameter of
            :class:`TorchTrainStep`.

        """
        pass

    def pre_val_batch(
        self, step: int, val_step: int, epoch: int, val_batch: Dict[str, Any]
    ) -> None:
        """
        Called right before a validation batch is processed.
        """
        pass

    def post_val_batch(
        self, step: int, val_step: int, epoch: int, val_batch_outputs: Dict[str, Any]
    ) -> None:
        """
        Called right after a validation batch is processed with the outputs of the batch.

        .. tip::
            This method can be used to modify ``val_batch_outputs`` in place, which is useful
            in scenarios like distributed training where you might need to aggregate metrics
            in a special way other than a simple average. If that's the case, make sure
            to set ``auto_aggregate_val_metric`` to ``False`` in :class:`TorchTrainStep`.

        """
        pass

    def post_val_loop(
        self, step: int, epoch: int, val_metric: float, best_val_metric: float
    ) -> None:
        """
        Called right after the validation loop finishes.
        """
        pass


@TrainCallback.register("torch::stop_early")
class StopEarlyCallback(TrainCallback):
    """
    A :class:`TrainCallback` for early stopping. Training is stopped early after
    ``patience`` steps without an improvement to the validation metric.

    .. tip::

        Registered as a :class:`TrainCallback` under the name "torch::stop_early".
    """

    def __init__(self, *args, patience: int = 10000, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.patience = patience
        self.best_step = 0

    def post_val_loop(
        self, step: int, epoch: int, val_metric: float, best_val_metric: float
    ) -> None:
        if val_metric == best_val_metric:
            self.best_step = step
        elif step > self.best_step + self.patience:
            raise StopEarly

    def state_dict(self) -> Dict[str, Any]:
        """
        Return any state that needs to be kept after a restart.
        """
        return {"patience": self.patience, "best_step": self.best_step}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load the state on a restart.
        """
        self.patience = state_dict["patience"]
        self.best_step = state_dict["best_step"]
