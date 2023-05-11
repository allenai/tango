import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.api import FullOptimStateDictConfig
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

from tango.common import Lazy, Tqdm
from tango.common.exceptions import ConfigurationError

from .fsdp_config import FSDPConfig
from .model import Model
from .optim import LRScheduler, Optimizer
from .train_config import TrainConfig
from .training_engine import TorchTrainingEngine, TrainingEngine


@TrainingEngine.register("torch::fsdp")
class FSDPTrainingEngine(TorchTrainingEngine):
    """
    A :class:`~tango.integrations.torch.TrainingEngine` that leverages Torch's
    :class:`~torch.distributed.fsdp.FullyShardedDataParallel` for use within
    :class:`~tango.integrations.torch.TorchTrainStep`.

    .. tip::
        Registered as an :class:`~tango.integrations.torch.TrainingEngine` under the name
        "torch::fsdp".

    .. tip::
        To get the best performance out of :class:`FSDPTrainingEngine` you should
        wrap individual layers of your model with :class:`~torch.distributed.fsdp.FullyShardedDataParallel`
        and/or :class:`~torch.distributed.algorithms._checkpoint.checkpoint_wrapper.checkpoint_wrapper`
        while instantiating them. You can use :class:`with_wrapped_modules()` to accomplish this.

    .. important::
        Only the parameters listed below should be defined in a configuration
        file. The other parameters will be automatically passed to the constructor
        within :class:`~tango.integrations.torch.TorchTrainStep`.

    .. warning::
        :class:`~FSDPTrainingEngine` can only be used in distributed training, i.e.
        when ``device_count > 1`` in the :class:`~tango.integrations.torch.TorchTrainStep`.

    For maximum memory savings, we recommend training with AMP enabled and the following
    :class:`FSDPConfig`:

    .. testcode::

        from tango.integrations.torch import FSDPConfig

        fsdp_config = FSDPConfig(
            reshard_after_forward=True,
            move_params_to_cpu=True,
            move_grads_to_cpu=True,
            mixed_precision=True,
        )

    For maximum training *speed*, we recommend training with AMP enabled and the following
    :class:`FSDPConfig`:

    .. testcode::

        from tango.integrations.torch import FSDPConfig

        fsdp_config = FSDPConfig(
            reshard_after_forward=False,
            move_params_to_cpu=False,
            move_grads_to_cpu=False,
            mixed_precision=True,
        )

    :param amp:
        Use automatic mixed precision (AMP). Default is ``False``.
    :param max_grad_norm:
        If set, gradients will be clipped to have this max norm. Default is ``None``.
    :param amp_use_bfloat16:
        Set to ``True`` to force using the ``bfloat16`` datatype in mixed precision training.
        Only applicable when ``amp=True``. If not specified, the default behavior will be
        to use ``bfloat16`` when training with AMP on CPU, otherwise not.
    :param fsdp_config:
        The options for :class:`~torch.distributed.fsdp.FullyShardedDataParallel`.
        If not specified, the default options will be used.

    """

    def __init__(
        self,
        train_config: TrainConfig,
        model: Lazy[Model],
        optimizer: Lazy[Optimizer],
        *,
        lr_scheduler: Optional[Lazy[LRScheduler]] = None,
        amp: bool = False,
        max_grad_norm: Optional[float] = None,
        amp_use_bfloat16: Optional[bool] = None,
        fsdp_config: Optional[FSDPConfig] = None,
    ) -> None:
        if not train_config.is_distributed:
            raise ConfigurationError(
                f"{self.__class__.__name__} can only be used with distributed training"
            )

        self.fsdp_config = fsdp_config or FSDPConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

        super().__init__(
            train_config,
            model,
            optimizer,
            lr_scheduler=lr_scheduler,
            amp=amp,
            max_grad_norm=max_grad_norm,
            amp_use_bfloat16=amp_use_bfloat16,
        )
        if amp:
            self.grad_scaler = ShardedGradScaler()

    def _construct_model(self, model: Union[Model, Lazy[Model]]) -> Model:
        if isinstance(model, Lazy):
            model = model.construct()
        if not self.fsdp_config.move_params_to_cpu:
            model.to(self.train_config.worker_local_default_device)
        return FSDP(model, **self.fsdp_config.as_kwargs())  # type: ignore

    def clip_grad_norm(self) -> None:
        if self.max_grad_norm is not None:
            self.model.clip_grad_norm_(self.max_grad_norm)  # type: ignore

    def get_model_state(self) -> Dict[str, torch.Tensor]:
        return self.model.state_dict()

    def load_model_state(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(state_dict)  # type: ignore

    def save_checkpoint(self, checkpoint_dir: Path, client_state: Dict[str, Any]) -> None:
        checkpoint_dir.mkdir(exist_ok=True)

        def save_state(state: Dict[str, Any], name: str):
            # only rank 0 writes any files
            if self.train_config.worker_id != 0:
                return

            temp_state_file = tempfile.NamedTemporaryFile(
                "w+b", dir=checkpoint_dir, delete=False, suffix=".pt"
            )
            try:
                with Tqdm.wrapattr(
                    temp_state_file,
                    "write",
                    desc=f"Saving {name} state",
                    leave=False,
                    disable=not self.train_config.is_local_main_process,
                ) as f:
                    torch.save(state, f)
                temp_state_file.close()
                os.replace(
                    temp_state_file.name,
                    checkpoint_dir / f"worker0_{name}.pt",
                )
            finally:
                if os.path.exists(temp_state_file.name):
                    os.remove(temp_state_file.name)

        with FSDP.state_dict_type(
            self.model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            FullOptimStateDictConfig(rank0_only=True, offload_to_cpu=True),
        ):
            save_state(self.get_model_state(), "model")
            save_state(FSDP.optim_state_dict(self.model, self.optimizer), "optimizer")
            if self.lr_scheduler is not None:
                save_state(self.lr_scheduler.state_dict(), "lr_scheduler")
            if self.grad_scaler is not None:
                save_state(self.grad_scaler.state_dict(), "grad_scaler")
            save_state(client_state, "trainer")

    def load_checkpoint(self, checkpoint_dir: Path) -> Dict[str, Any]:
        with FSDP.state_dict_type(
            self.model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            FullOptimStateDictConfig(rank0_only=True, offload_to_cpu=True),
        ):
            if self.train_config.worker_id == 0:
                model_state_dict = torch.load(checkpoint_dir / "worker0_model.pt")
                optimizer_state_dict = torch.load(checkpoint_dir / "worker0_optimizer.pt")
            else:
                model_state_dict = {}
                optimizer_state_dict = {}

            self.load_model_state(model_state_dict)
            optimizer_state_dict = FSDP.optim_state_dict_to_load(
                optimizer_state_dict, self.model, self.optimizer
            )
            self.optimizer.load_state_dict(optimizer_state_dict)

            # The states for LR scheduler, grad scaler, and trainer are identical on all workers, so we have
            # all of them load the same files.
            if self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(
                    torch.load(checkpoint_dir / "worker0_lr_scheduler.pt")
                )
            if self.grad_scaler is not None:
                self.grad_scaler.load_state_dict(
                    torch.load(checkpoint_dir / "worker0_grad_scaler.pt")
                )

            return torch.load(checkpoint_dir / "worker0_trainer.pt")
