from pathlib import Path
from typing import Any, Dict

import torch

from tango.integrations.torch import Accelerator


@Accelerator.register("deepspeed")
class DeepSpeedAccelerator(Accelerator):
    def forward_train(
        self, micro_batch: Dict[str, Any], micro_batch_idx: int, num_micro_batches: int
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward_eval(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def backward(self, loss: torch.Tensor) -> None:
        raise NotImplementedError

    def step(self) -> None:
        raise NotImplementedError

    def save_checkpoint(self, path: Path, client_state: Dict[str, Any]) -> None:
        raise NotImplementedError

    def load_checkpoint(self, path: Path) -> Dict[str, Any]:
        raise NotImplementedError

    def save_complete_weights_from_checkpoint(
        self, checkpoint_path: Path, weights_path: Path
    ) -> None:
        raise NotImplementedError
