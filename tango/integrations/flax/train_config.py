from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class TrainConfig:
    """
    Encapsulates the parameters of :class:`FlaxTrainStep`. This is used to pass all the training
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

    step_name: Optional[str] = None
    """
    The name of the current step.
    """

    train_split: Optional[str] = "train"
    """
    The name of the training split.
    """

    validation_split: Optional[str] = None
    """
    The name of the validation split.
    """

    seed: int = 42
    """
    The random seed used to generate
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
