from typing import Union, Optional

import transformers
from transformers import SchedulerType

from tango.integrations.torch import Optimizer, LRScheduler

Optimizer.register("transformers::adamw")(transformers.optimization.AdamW)
Optimizer.register("transformers::adafactor")(transformers.optimization.Adafactor)

LRScheduler.register("transformers::adafactor")(transformers.optimization.AdafactorSchedule)


class TransformersLRScheduler(LRScheduler):
    def __init__(self, **kwargs):
        raise NotImplementedError(
            "This should not be called directly. This class is just a holder for the"
            "initialization functions below."
        )

    @classmethod
    def get_constant_schedule(cls, optimizer: Optimizer, last_epoch: int = -1):
        """
        Create a schedule with a constant learning rate, using the learning rate set in optimizer.

        Args:
            optimizer ([`~torch.optim.Optimizer`]):
                The optimizer for which to schedule the learning rate.
            last_epoch (`int`, *optional*, defaults to -1):
                The index of the last epoch when resuming training.

        Return:
            `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
        """
        return transformers.optimization.get_constant_schedule(optimizer, last_epoch)

    @classmethod
    def get_constant_schedule_with_warmup(
        cls, optimizer: Optimizer, num_warmup_steps: int, last_epoch: int = -1
    ):
        """
        Create a schedule with a constant learning rate preceded by a warmup period during which the learning rate
        increases linearly between 0 and the initial lr set in the optimizer.

        Args:
            optimizer ([`~torch.optim.Optimizer`]):
                The optimizer for which to schedule the learning rate.
            num_warmup_steps (`int`):
                The number of steps for the warmup phase.
            last_epoch (`int`, *optional*, defaults to -1):
                The index of the last epoch when resuming training.

        Return:
            `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
        """
        return transformers.optimization.get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps, last_epoch
        )

    @classmethod
    def get_linear_schedule_with_warmup(
        cls, optimizer, num_warmup_steps: int, num_training_steps: int, last_epoch: int = -1
    ):
        """
        Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
        a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

        Args:
            optimizer ([`~torch.optim.Optimizer`]):
                The optimizer for which to schedule the learning rate.
            num_warmup_steps (`int`):
                The number of steps for the warmup phase.
            num_training_steps (`int`):
                The total number of training steps.
            last_epoch (`int`, *optional*, defaults to -1):
                The index of the last epoch when resuming training.

        Return:
            `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
        """
        return transformers.optimization.get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps, last_epoch
        )

    @classmethod
    def get_cosine_schedule_with_warmup(
        cls,
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
    ):
        """
        Create a schedule with a learning rate that decreases following the values of the cosine function between the
        initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
        initial lr set in the optimizer.

        Args:
            optimizer ([`~torch.optim.Optimizer`]):
                The optimizer for which to schedule the learning rate.
            num_warmup_steps (`int`):
                The number of steps for the warmup phase.
            num_training_steps (`int`):
                The total number of training steps.
            num_cycles (`float`, *optional*, defaults to 0.5):
                The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
                following a half-cosine).
            last_epoch (`int`, *optional*, defaults to -1):
                The index of the last epoch when resuming training.

        Return:
            `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
        """
        return transformers.optimization.get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps, num_cycles, last_epoch
        )

    @classmethod
    def get_cosine_with_hard_restarts_schedule_with_warmup(
        cls,
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: int = 1,
        last_epoch: int = -1,
    ):
        """
        Create a schedule with a learning rate that decreases following the values of the cosine function between the
        initial lr set in the optimizer to 0, with several hard restarts, after a warmup period during which it increases
        linearly between 0 and the initial lr set in the optimizer.

        Args:
            optimizer ([`~torch.optim.Optimizer`]):
                The optimizer for which to schedule the learning rate.
            num_warmup_steps (`int`):
                The number of steps for the warmup phase.
            num_training_steps (`int`):
                The total number of training steps.
            num_cycles (`int`, *optional*, defaults to 1):
                The number of hard restarts to use.
            last_epoch (`int`, *optional*, defaults to -1):
                The index of the last epoch when resuming training.

        Return:
            `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
        """
        return transformers.optimization.get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps, num_cycles, last_epoch
        )

    @classmethod
    def get_polynomial_decay_schedule_with_warmup(
        cls,
        optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        lr_end=1e-7,
        power=1.0,
        last_epoch=-1,
    ):
        """
        Create a schedule with a learning rate that decreases as a polynomial decay from the initial lr set in the
        optimizer to end lr defined by *lr_end*, after a warmup period during which it increases linearly from 0 to the
        initial lr set in the optimizer.

        Args:
            optimizer ([`~torch.optim.Optimizer`]):
                The optimizer for which to schedule the learning rate.
            num_warmup_steps (`int`):
                The number of steps for the warmup phase.
            num_training_steps (`int`):
                The total number of training steps.
            lr_end (`float`, *optional*, defaults to 1e-7):
                The end LR.
            power (`float`, *optional*, defaults to 1.0):
                Power factor.
            last_epoch (`int`, *optional*, defaults to -1):
                The index of the last epoch when resuming training.

        Note: *power* defaults to 1.0 as in the fairseq implementation, which in turn is based on the original BERT
        implementation at
        https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/optimization.py#L37

        Return:
            `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.

        """
        return transformers.optimization.get_polynomial_decay_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps, lr_end, power, last_epoch
        )

    @classmethod
    def get_scheduler(
        cls,
        name: Union[str, SchedulerType],
        optimizer: Optimizer,
        num_warmup_steps: Optional[int] = None,
        num_training_steps: Optional[int] = None,
    ):
        """
        Unified API to get any scheduler from its name.

        Args:
            name (`str` or `SchedulerType`):
                The name of the scheduler to use.
            optimizer (`torch.optim.Optimizer`):
                The optimizer that will be used during training.
            num_warmup_steps (`int`, *optional*):
                The number of warmup steps to do. This is not required by all schedulers (hence the argument being
                optional), the function will raise an error if it's unset and the scheduler type requires it.
            num_training_steps (`int``, *optional*):
                The number of training steps to do. This is not required by all schedulers (hence the argument being
                optional), the function will raise an error if it's unset and the scheduler type requires it.
        """
        return transformers.optimization.get_scheduler(
            name, optimizer, num_warmup_steps, num_training_steps
        )
