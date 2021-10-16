from pathlib import Path
from typing import Optional, Dict, Any, List

import torch

from .data import DataLoader
from .model import Model
from .optim import Optimizer, LRScheduler
from tango.common.registrable import Registrable


class TrainCallback(Registrable):
    """
    A ``TrainCallback`` is a :class:`~tango.common.registrable.Registrable` class
    that can be used within :class:`TorchTrainStep` to customize behavior in the training
    loop.

    .. tip::
        All of the parameters to this base class will be automatically set within
        the training loop, so you shouldn't include them in your config for your callbacks.

    .. seealso::
        See :class:`~tango.integrations.wandb.WandbTrainCallback` for an example
        implementation.

    """

    def __init__(
        self,
        work_dir: Path,
        model: Model,
        optimizer: Optimizer,
        train_dataloader: DataLoader,
        validation_dataloader: Optional[DataLoader] = None,
        lr_scheduler: Optional[LRScheduler] = None,
        is_local_main_process: bool = True,
        world_size: int = 1,
        worker_id: int = 0,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.work_dir = work_dir
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.lr_scheduler = lr_scheduler
        self.is_local_main_process = is_local_main_process
        self.worker_id = worker_id
        self.world_size = world_size
        self.device = device

    def state_dict(self) -> Dict[str, Any]:
        """
        Return any state that needs to be kept after a restart.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load the state on a restart.
        """
        pass

    def pre_train_loop(self) -> None:
        """
        Called right before the first batch is processed, or after a restart.
        """
        pass

    def post_train_loop(self) -> None:
        """
        Called after the training loop completes.

        This is the last method that is called, so any cleanup can be done in this method.
        """
        pass

    def pre_checkpoint(self, checkpoint_state: Dict[str, Any]) -> None:
        """
        Called directly before the checkpoint is saved.
        """
        pass

    def post_checkpoint(self, checkpoint_path: Path) -> None:
        """
        Called directly after a checkpoint is saved.
        """
        pass

    def pre_batch(self, step: int, batch: List[Dict[str, Any]]) -> None:
        """
        Called directly before processing a batch.

        .. note::
            A type of ``batch`` is a list because with gradient accumulation there will
            more than one "micro batch" in the batch.

        """
        pass

    def post_batch(self, step: int, batch_loss: float) -> None:
        """
        Called directly after processing a batch, but before unscaling gradients,
        clipping gradients, and taking an optimizer step.

        .. note::
            The ``batch_loss`` here is the loss local to the current worker, not the
            overall (average) batch loss across distributed workers.

        """
        pass

    def pre_log_batch(self, step: int, metrics_to_log: Dict[str, float]) -> None:
        """
        Called right before ``metrics_to_log`` are logged to the progress bar.

        .. warning::
            This may be called twice for a given ``step``: once before the validation
            loop and once right after the validation loop with the updated validation metric.

            Therefore if you're using your callback to log metrics externally,
            it may be better to use the :meth:`post_batch()` and :meth:`post_val_loop()`
            methods instead.

        """
        pass

    def pre_val_batch(self, step: int, val_step: int, val_batch: Dict[str, Any]) -> None:
        """
        Called right before a validation batch is processed.
        """
        pass

    def post_val_batch(self, step: int, val_step: int, val_batch_outputs: Dict[str, Any]) -> None:
        """
        Called right after a validation batch is processed with the outputs of the batch.
        """
        pass

    def post_val_loop(self, step: int, val_metric_name: str, val_metric: float) -> None:
        """
        Called right after the validation loop finishes.
        """
        pass
