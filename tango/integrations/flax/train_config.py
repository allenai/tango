from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import jax


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

    # type?
    # params

    step_name: Optional[str] = None
    """
    The name of the current step.
    """

    # worker_id

    train_split: str = "train"
    """
    The name of the training split.
    """

    validation_split: Optional[str] = None
    """
    The name of the validation split.
    """

    seed: Union[Any, jax.random.PRNGKeyArray] = jax.random.PRNGKey(10)
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

    def should_log_this_step(self, step: int) -> bool:
        raise NotImplementedError
