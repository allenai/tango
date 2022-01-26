from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP

from tango.common import Lazy
from tango.common.exceptions import ConfigurationError
from tango.integrations.torch import (
    Accelerator,
    LRScheduler,
    Model,
    Optimizer,
    TorchAccelerator,
    TrainConfig,
)

if TYPE_CHECKING:
    from collections import OrderedDict  # noqa: F401


@Accelerator.register("fairscale")
class FairScaleAccelerator(TorchAccelerator):
    """
    An :class:`~tango.integrations.torch.Accelerator` that leverages FairScale's
    :class:`~fairscale.nn.FullyShardedDataParallel`.

    .. tip::
        Registered as an :class:`~tango.integrations.torch.Accelerator` under the name
        "fairscale".
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
        reshard_after_forward: bool = True,
    ) -> None:
        if not train_config.is_distributed:
            raise ConfigurationError(
                f"{self.__class__.__name__} can only be used with distributed training"
            )
        super().__init__(
            train_config,
            model,
            optimizer,
            lr_scheduler=lr_scheduler,
            amp=amp,
            max_grad_norm=max_grad_norm,
        )
        self.reshard_after_forward = reshard_after_forward
        self.mixed_precision = amp

    def _construct_model(self, model: Lazy[Model]) -> Model:
        model: Model = model.construct()
        model.to(self.train_config.worker_local_default_device)
        return FSDP(
            model,
            reshard_after_forward=self.reshard_after_forward,
            mixed_precision=self.mixed_precision,
        )

    def clip_grad_norm(self) -> None:
        if self.max_grad_norm is not None:
            self.model.clip_grad_norm_(self.max_grad_norm)  # type: ignore

    def get_model_state(self) -> "OrderedDict[str, torch.Tensor]":
        return self.model.local_state_dict()  # type: ignore

    def load_model_state(self, state_dict: "OrderedDict[str, torch.Tensor]") -> None:
        self.model.load_local_state_dict(state_dict)  # type: ignore

    def save_complete_weights_from_checkpoint(
        self, checkpoint_path: Path, weights_path: Path
    ) -> None:
        sharded_weights: List[Dict[str, torch.Tensor]] = []
        sharded_metadata: List[Dict[str, Any]] = []
        for path in checkpoint_path.resolve().glob("worker*_model.pt"):
            sharded_state = torch.load(path, map_location="cpu")
            sharded_weights.append(sharded_state["weights"])
            sharded_metadata.append(sharded_state["metadata"])
        full_state = FSDP.consolidate_shard_weights(sharded_weights, sharded_metadata)
        del sharded_weights
        del sharded_metadata
        torch.save(full_state, weights_path)
