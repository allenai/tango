from pathlib import Path
from typing import Any, Dict, Optional, Union

import deepspeed
import torch
from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict

from tango.common import Lazy
from tango.integrations.torch import (
    Accelerator,
    LRScheduler,
    Model,
    Optimizer,
    TrainConfig,
)


@Accelerator.register("deepspeed")
class DeepSpeedAccelerator(Accelerator):
    def __init__(
        self,
        train_config: TrainConfig,
        model: Lazy[Model],
        optimizer: Lazy[Optimizer],
        deepspeed_config: Union[Path, Dict[str, Any]],
        *,
        lr_scheduler: Optional[Lazy[LRScheduler]] = None,
    ) -> None:
        super().__init__(train_config, model, optimizer, lr_scheduler=lr_scheduler)
        self.train_engine, self.optimizer, _, self.lr_scheduler = deepspeed.initialize(
            model=self.model,
            model_parameters=self.model.parameters(),
            lr_scheduler=lr_scheduler,
            dist_init_required=self.train_config.is_distributed,
            config=deepspeed_config,
        )

    def forward_train(
        self, micro_batch: Dict[str, Any], micro_batch_idx: int, num_micro_batches: int
    ) -> torch.Tensor:
        outputs = self.train_engine(micro_batch)
        return outputs["loss"] / num_micro_batches

    def forward_eval(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return self.train_engine(batch)

    def backward(self, loss: torch.Tensor) -> None:
        self.train_engine.backward(loss)

    def step(self) -> None:
        self.train_engine.step()

    def save_checkpoint(self, path: Path, client_state: Dict[str, Any]) -> None:
        self.train_engine.save_checkpoint(
            path, client_state["training_steps"], client_save=client_state
        )

    def load_checkpoint(self, path: Path) -> Dict[str, Any]:
        _, client_state = self.train_engine.load_checkpoint(path)
        return client_state

    def save_complete_weights_from_checkpoint(
        self, checkpoint_path: Path, weights_path: Path
    ) -> None:
        if self.train_engine.zero_optimization():
            return convert_zero_checkpoint_to_fp32_state_dict(checkpoint_path, weights_path)
        else:
            raise NotImplementedError
