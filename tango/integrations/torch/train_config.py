from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


@dataclass
class TrainConfig:
    """
    Encapsulates the parameters of :class:`TorchTrainStep`. This is used to pass all the training
    options to :class:`TrainCallback`.
    """

    step_id: str
    """
    The unique ID of the current step.
    """

    work_dir: Path
    """
    The working directory for the training run.
    """

    worker_id: int = 0
    """
    The ID of the distributed worker.
    """

    train_split: str = "train"
    """
    The name of the training split.
    """

    validation_split: Optional[str] = None
    """
    The name of the validation split.
    """

    seed: int = 42
    """
    The random seed.
    """

    train_steps: Optional[int] = None
    """
    The number of steps to train for.
    """

    train_epochs: Optional[int] = None
    """
    The number of epochs to train for.

    You cannot specify `train_steps` and `train_epochs` at the same time.
    """

    validation_steps: Optional[int] = None
    """
    The number of validation steps.
    """

    grad_accum: int = 1
    """
    The number of micro-batches per gradient accumulation mini-batch.
    """

    log_every: int = 10
    """
    Controls the frequency of log updates.
    """

    checkpoint_every: int = 100
    """
    Controls the frequency of checkpoints.
    """

    validate_every: Optional[int] = None
    """
    Controls the frequency of the validation loop.
    """

    is_distributed: bool = False
    """
    Whether or not the training job is distributed.
    """

    devices: Optional[List[int]] = None
    """
    The devices used (for distributed jobs).
    """

    distributed_address: str = "127.0.0.1"
    """
    The IP address of the main distributed process.
    """

    distributed_port: int = 54761
    """
    The port of the main distributed process.
    """

    val_metric_name: str = "loss"
    """
    The name of the validation metric to track.
    """

    minimize_val_metric: bool = True
    """
    Should be ``True`` when the validation metric being tracked should be minimized.
    """

    auto_aggregate_val_metric: bool = True
    """
    Controls automatic aggregation of validation metric.
    """

    remove_stale_checkpoints: bool = True
    """
    Controls removal of stale checkpoints.
    """

    world_size: int = 1
    """
    The number of distributed workers.
    """

    _worker_local_default_device: Optional[torch.device] = None

    _device_type: Optional[str] = None  # either "cuda" or "cpu"

    @property
    def worker_local_default_device(self) -> torch.device:
        """
        The default ``torch`` device for the current worker.
        """
        if self._worker_local_default_device is not None:
            return self._worker_local_default_device
        else:
            if self.devices:
                device_id = self.devices[self.worker_id]
                if device_id >= 0:
                    device = torch.device(f"cuda:{device_id}")
                else:
                    device = torch.device("cpu")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
            self._worker_local_default_device = device
            return device

    @property
    def device_type(self) -> str:
        if self._device_type is None:
            device_type = (
                "cpu" if self.worker_local_default_device == torch.device("cpu") else "cuda"
            )
            self._device_type = device_type
            return device_type
        else:
            return self._device_type

    @property
    def is_local_main_process(self) -> bool:
        """
        Whether the local process is the main distributed worker.
        """
        return self.worker_id == 0

    @property
    def state_path(self) -> Path:
        """
        The path to the latest state checkpoint file.
        """
        return self.work_dir / "checkpoint_state_latest"

    @property
    def best_state_path(self) -> Path:
        """
        The path to the best state checkpoint file according to the validation metric or training
        loss (if no validation split is given).
        """
        return self.work_dir / "checkpoint_state_best"

    def state_path_for_step(self, step: int) -> Path:
        return self.work_dir / f"checkpoint_state_step{step + 1}"

    @property
    def final_weights_path(self) -> Path:
        return self.work_dir / "weights.pt"

    def should_log_this_step(self, step: int) -> bool:
        assert self.train_steps is not None
        return step == 0 or (step + 1) % self.log_every == 0 or step == self.train_steps - 1

    def should_checkpoint_this_step(self, step: int) -> bool:
        assert self.train_steps is not None
        return ((step + 1) % self.checkpoint_every == 0) or step == self.train_steps - 1

    def should_log_this_val_step(self, val_step: int) -> bool:
        assert self.validation_steps is not None
        return val_step % self.log_every == 0 or val_step == self.validation_steps - 1

    def as_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if not k.startswith("_")}
