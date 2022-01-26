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
    TorchTrainEngine,
    TrainConfig,
    TrainEngine,
)

from .fsdp_config import FSDPConfig


@TrainEngine.register("fairscale")
class FairScaleTrainEngine(TorchTrainEngine):
    """
    A :class:`~tango.integrations.torch.TrainEngine` that leverages FairScale's
    :class:`~fairscale.nn.FullyShardedDataParallel` for use within
    :class:`~tango.integrations.torch.TorchTrainStep`.

    .. tip::
        Registered as an :class:`~tango.integrations.torch.TrainEngine` under the name
        "fairscale".

    .. tip::
        To get the best performance out of :class:`FairScaleTrainEngine` you should
        wrap individual layers of your model with :class:`~fairscale.nn.FullyShardedDataParallel`
        while instantiating them, such as in the example in the FairScale docs and in the
        `language model example </examples/train_lm.html>`_.

    .. important::
        Only the parameters listed below should be defined in a configuration
        file. The other parameters will be automatically passed to the constructor
        within :class:`~tango.integrations.torch.TorchTrainStep`.

    Parameters
    ----------
    amp : :class:`bool`, optional
        Use automatic mixed precision. Default is ``False``.
    max_grad_norm : :class:`float`, optional
        If set, gradients will be clipped to have this max norm. Default is ``None``.
    fsdp_config : :class:`FSDPConfig`
        The options for :class:`~fairscale.nn.FullyShardedDataParallel`.

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
        fsdp_config: Optional[FSDPConfig] = None,
    ) -> None:
        if not train_config.is_distributed:
            raise ConfigurationError(
                f"{self.__class__.__name__} can only be used with distributed training"
            )

        self.fsdp_config = fsdp_config or FSDPConfig()
        self.fsdp_config.mixed_precision = amp

        super().__init__(
            train_config,
            model,
            optimizer,
            lr_scheduler=lr_scheduler,
            amp=amp,
            max_grad_norm=max_grad_norm,
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
