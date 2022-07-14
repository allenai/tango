import logging
from pathlib import Path
from typing import Any, Dict, Optional

from tango.common.dataset_dict import DatasetDictBase
from tango.common.registrable import Registrable
from tango.workspace import Workspace

from .data import DataLoader
from .model import Model
from .optim import Optimizer
from .train_config import TrainConfig


class TrainCallback(Registrable):
    """ """

    def __init__(
        self,
        workspace: Workspace,
        train_config: TrainConfig,
        dataset: DatasetDictBase,
        train_dataloader: DataLoader,
        model: Model,
        optimizer: Optimizer,
        validation_dataloader: Optional[DataLoader] = None,
    ) -> None:
        self.workspace = workspace
        self.train_config = train_config
        self.dataset = dataset
        self.train_dataloader = train_dataloader
        self.model = model
        self.optimizer = optimizer
        self.validation_dataloader = validation_dataloader
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    def step_id(self) -> str:
        """
        The unique ID of the current :class:`~tango.Step`.
        """
        return self.train_config.step_id

    @property
    def step_name(self) -> Optional[str]:
        """
        The name of the current:class:`~tango.Step`.
        """
        return self.train_config.step_name

    @property
    def work_dir(self) -> Path:
        """
        The working directory of the current train step
        """
        return self.train_config.work_dir

    @property
    def model(self) -> Model:
        """
        The :class:`Model` being trained
        """
        raise NotImplementedError

    def state_dict(self) -> Dict[str, Any]:
        """
        Return any state that needs to be kept after a restart.

        Some callbacks need to maintain state across restarts. This is the callback's opportunity to
        save it's state. It will be restored using :meth:`load_state_dict`.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Load the state on a restart.

        Some callbacks need to maintain state across restarts. This is the callback's opportunity to
        restore it's state. It gets saved using :meth:`state_dict`.
        """
        pass

    def pre_train_loop(self) -> None:
        """
        Called right before the first batch is processed, or after a restart
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
        Called before start of an epoch. Epochs start at 0.
        """
        pass

    def post_epoch(self, step: int, epoch: int) -> None:
        """
        Called after an epoch is completed. Epochs start at 0.
        """
        pass

    # what will be the type for batch?
    def pre_batch(self, step: int, epoch: int, batch) -> None:
        """
        Called directly before processing a batch.
        """

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
            :class:`FlaxTrainStep`.

        """
        pass

    # type for val_batch?
    def pre_val_batch(self, step: int, val_step: int, epoch: int, val_batch) -> None:
        """
        Called right before a validation batch is processed.
        """

    # type for val_batch_outputs?
    def post_val_batch(self, step: int, val_step: int, epoch: int, val_batch_outputs) -> None:
        """
        Called right after a validation batch is processed with the outputs of the batch.

        .. tip::
            This method can be used to modify ``val_batch_outputs`` in place, which is useful
            in scenarios like distributed training where you might need to aggregate metrics
            in a special way other than a simple average. If that's the case, make sure
            to set ``auto_aggregate_val_metric`` to ``False`` in :class:`FlaxTrainStep`.

        """
        pass

    # should we have a log_val_batch() over here?

    def post_val_loop(
        self, step: int, epoch: int, val_metric: Optional[float], best_val_metric: Optional[float]
    ) -> None:
        """
        Called right after the evaluation loop finishes
        """
        pass
