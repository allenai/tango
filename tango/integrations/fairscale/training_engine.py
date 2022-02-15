import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.optim.grad_scaler import ShardedGradScaler

from tango.common import Lazy
from tango.common.exceptions import ConfigurationError
from tango.integrations.torch import (
    LRScheduler,
    Model,
    Optimizer,
    TorchTrainingEngine,
    TrainConfig,
    TrainingEngine,
)

from .fsdp_config import FSDPConfig


@TrainingEngine.register("fairscale")
class FairScaleTrainingEngine(TorchTrainingEngine):
    """
    A :class:`~tango.integrations.torch.TrainingEngine` that leverages FairScale's
    :class:`~fairscale.nn.FullyShardedDataParallel` for use within
    :class:`~tango.integrations.torch.TorchTrainStep`.

    .. tip::
        Registered as an :class:`~tango.integrations.torch.TrainingEngine` under the name
        "fairscale".

    .. tip::
        To get the best performance out of :class:`FairScaleTrainingEngine` you should
        wrap individual layers of your model with :class:`~fairscale.nn.FullyShardedDataParallel`
        and/or :class:`~fairscale.nn.checkpoint.checkpoint_wrapper`
        while instantiating them. You can use :class:`with_wrapped_modules()` to accomplish this.

    .. important::
        Only the parameters listed below should be defined in a configuration
        file. The other parameters will be automatically passed to the constructor
        within :class:`~tango.integrations.torch.TorchTrainStep`.

    .. warning::
        :class:`~FairScaleTrainingEngine` can only be used in distributed training, i.e.
        when ``device_count > 1`` in the :class:`~tango.integrations.torch.TorchTrainStep`.

    For maximum memory savings, we recommend training with AMP enabled and the following
    :class:`FSDPConfig`:

    .. testcode::

        from tango.integrations.fairscale import FSDPConfig

        fsdp_config = FSDPConfig(
            reshard_after_forward=True,
            move_params_to_cpu=True,
            move_grads_to_cpu=True,
            mixed_precision=True,
        )

    For maximum training *speed*, we recommend training with AMP enabled and the following
    :class:`FSDPConfig`:

    .. testcode::

        from tango.integrations.fairscale import FSDPConfig

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
        The options for :class:`~fairscale.nn.FullyShardedDataParallel`.
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

    def _construct_model(self, model: Lazy[Model]) -> Model:
        model: Model = model.construct()
        if not self.fsdp_config.move_params_to_cpu:
            model.to(self.train_config.worker_local_default_device)
        return FSDP(model, **self.fsdp_config.as_kwargs())

    def clip_grad_norm(self) -> None:
        if self.max_grad_norm is not None:
            self.model.clip_grad_norm_(self.max_grad_norm)  # type: ignore

    def get_model_state(self) -> Dict[str, torch.Tensor]:
        return {
            "weights": self.model.local_state_dict(),  # type: ignore
            "metadata": self.model.local_metadata_dict(),  # type: ignore
        }

    def load_model_state(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.model.load_local_state_dict(state_dict["weights"])  # type: ignore

    def save_complete_weights_from_checkpoint(
        self, checkpoint_dir: Path, weights_path: Path
    ) -> None:
        self.logger.info("Consolidating sharded checkpoint weights...")
        sharded_weights: List[Dict[str, torch.Tensor]] = []
        sharded_metadata: List[Dict[str, Any]] = []
        for path in checkpoint_dir.resolve().glob("worker*_model.pt"):
            sharded_state = torch.load(path, map_location="cpu")
            sharded_weights.append(sharded_state["weights"])
            sharded_metadata.append(sharded_state["metadata"])
        full_state = FSDP.consolidate_shard_weights(sharded_weights, sharded_metadata)
        del sharded_weights
        del sharded_metadata
        torch.save(full_state, weights_path)
