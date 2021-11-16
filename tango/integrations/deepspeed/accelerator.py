import os
from pathlib import Path
from typing import Any, Dict, Optional

import deepspeed
import torch
import torch.distributed as dist
from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict
from overrides import overrides

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
        deepspeed_config: Dict[str, Any],
        *,
        lr_scheduler: Optional[Lazy[LRScheduler]] = None,
    ) -> None:
        self.device = train_config.worker_local_default_device
        if train_config.is_distributed:
            # Initialize distributed process group.
            backend: str
            if self.device != torch.device("cpu"):
                torch.cuda.set_device(self.device)
                backend = "nccl"
            else:
                backend = "gloo"
            dist.init_process_group(
                backend=backend,
                init_method=f"tcp://{train_config.distributed_address}:{train_config.distributed_port}",
                world_size=train_config.world_size,
                rank=train_config.worker_id,
            )
            os.environ["RANK"] = str(train_config.worker_id)
            os.environ["LOCAL_RANK"] = str(train_config.worker_id)
            os.environ["WORLD_SIZE"] = str(train_config.world_size)
            os.environ["MASTER_ADDR"] = str(train_config.distributed_address)
            os.environ["MASTER_PORT"] = str(train_config.distributed_port)

        super().__init__(train_config, model, optimizer, lr_scheduler=lr_scheduler)

        # Make sure deepspeed config has everything it needs.
        deepspeed_config["train_micro_batch_size_per_gpu"] = 1
        deepspeed_config["gradient_accumulation_steps"] = self.train_config.grad_accum

        # Initialize deepspeed engine.
        self.train_engine, self.optimizer, _, self.lr_scheduler = deepspeed.initialize(
            model=self.model,
            model_parameters=self.model.parameters(),
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            dist_init_required=False,
            config=deepspeed_config,
        )

    @overrides
    def _construct_model(self, model: Lazy[Model]) -> Model:
        with deepspeed.zero.Init():
            model: Model = model.construct()  # type: ignore[no-redef]
        return model.to(self.train_config.worker_local_default_device)  # type: ignore[attr-defined]

    @overrides
    def _construct_optimizer(self, optimizer: Lazy[Optimizer]) -> Optimizer:
        optimizer: Optimizer = optimizer.construct(
            params=self.model.parameters(), model_params=self.model.parameters()
        )
        return optimizer

    def forward_train(
        self, micro_batch: Dict[str, Any], micro_batch_idx: int, num_micro_batches: int
    ) -> torch.Tensor:
        micro_batch = self._move_to_device(micro_batch, self.device)
        outputs = self.train_engine(**micro_batch)
        return outputs["loss"] / num_micro_batches

    def forward_eval(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        batch = self._move_to_device(batch, self.device)
        return self.train_engine(**batch)

    def backward(self, loss: torch.Tensor) -> None:
        self.train_engine.backward(loss)

    def step(self) -> None:
        self.train_engine.step()

    def save_checkpoint(self, path: Path, client_state: Dict[str, Any]) -> None:
        self.train_engine.save_checkpoint(
            path, client_state["training_steps"], client_state=client_state
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
