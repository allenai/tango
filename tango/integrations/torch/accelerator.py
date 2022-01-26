import os
import tempfile
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, cast

import torch
import torch.distributed as dist
import torch.nn as nn

from tango.common import Lazy, Registrable, Tqdm

from .model import Model
from .optim import LRScheduler, Optimizer
from .train_config import TrainConfig

if TYPE_CHECKING:
    from collections import OrderedDict  # noqa: F401


class Accelerator(Registrable):
    default_implementation = "torch"

    def __init__(
        self,
        train_config: TrainConfig,
        model: Lazy[Model],
        optimizer: Lazy[Optimizer],
        *,
        lr_scheduler: Optional[Lazy[LRScheduler]] = None,
    ) -> None:
        self.train_config = train_config
        self.model = self._construct_model(model)
        self.optimizer = self._construct_optimizer(optimizer)
        self.lr_scheduler: Optional[LRScheduler] = None
        if lr_scheduler is not None:
            self.lr_scheduler = self._construct_lr_scheduler(lr_scheduler)

    def _construct_model(self, model: Lazy[Model]) -> Model:
        model: Model = model.construct()
        return model.to(self.train_config.worker_local_default_device)

    def _construct_optimizer(self, optimizer: Lazy[Optimizer]) -> Optimizer:
        optimizer: Optimizer = optimizer.construct(params=self.model.parameters())
        return optimizer

    def _construct_lr_scheduler(self, lr_scheduler: Lazy[LRScheduler]) -> LRScheduler:
        lr_scheduler: LRScheduler = lr_scheduler.construct(optimizer=self.optimizer)
        return lr_scheduler

    @classmethod
    def _move_to_device(cls, o: Any, device: torch.device) -> Any:
        if isinstance(o, torch.Tensor):
            return o.to(device)
        elif isinstance(o, dict):
            return {k: cls._move_to_device(v, device) for k, v in o.items()}
        elif isinstance(o, list):
            return [cls._move_to_device(x, device) for x in o]
        elif isinstance(o, tuple):
            return tuple((cls._move_to_device(x, device) for x in o))
        else:
            return o

    @abstractmethod
    def forward_train(
        self, micro_batch: Dict[str, Any], micro_batch_idx: int, num_micro_batches: int
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward_eval(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def backward(self, loss: torch.Tensor) -> None:
        raise NotImplementedError

    @abstractmethod
    def step(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self, path: Path, client_state: Dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_checkpoint(self, path: Path) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def save_complete_weights_from_checkpoint(
        self, checkpoint_path: Path, weights_path: Path
    ) -> None:
        raise NotImplementedError


@Accelerator.register("torch")
class TorchAccelerator(Accelerator):
    """
    This accelerator only uses native PyTorch functionality to provide
    vanilla distributed data parallel training and AMP.

    Parameters
    ----------
    amp : :class:`bool`, optional
        Use automatic mixed precision. Default is ``False``.
    max_grad_norm : :class:`float`, optional
        If set, gradients will be clipped to have this max norm. Default is ``None``.
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
    ) -> None:
        self.amp = amp
        self.max_grad_norm = max_grad_norm
        self.grad_scaler: Optional[torch.cuda.amp.GradScaler] = (
            None if not amp else torch.cuda.amp.GradScaler()
        )
        self.device = self.train_config.worker_local_default_device
        if self.train_config.is_distributed:
            # Initialize distributed process group.
            backend: str
            if self.device != torch.device("cpu"):
                torch.cuda.set_device(self.device)
                backend = "nccl"
            else:
                backend = "gloo"
            dist.init_process_group(
                backend=backend,
                init_method=f"tcp://{self.train_config.distributed_address}:{self.train_config.distributed_port}",
                world_size=self.train_config.world_size,
                rank=self.train_config.worker_id,
            )

        super().__init__(train_config, model, optimizer, lr_scheduler=lr_scheduler)

    def _construct_model(self, model: Lazy[Model]) -> Model:
        model: Model = model.construct()
        model.to(self.train_config.worker_local_default_device)
        # Wrap model with DDP wrapper.
        if self.train_config.is_distributed:
            model = cast(Model, nn.parallel.DistributedDataParallel(model))
        return model

    def forward_train(
        self, micro_batch: Dict[str, Any], micro_batch_idx: int, num_micro_batches: int
    ) -> torch.Tensor:
        if micro_batch_idx == 0:
            self.optimizer.zero_grad(set_to_none=True)

        # Move tensors to right device.
        micro_batch = self._move_to_device(micro_batch, self.device)

        with torch.cuda.amp.autocast(enabled=self.amp):
            outputs = self.model(**micro_batch)
            micro_batch_loss = outputs["loss"] / num_micro_batches

        return micro_batch_loss

    def forward_eval(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        # Move tensors to right device.
        batch = self._move_to_device(batch, self.device)

        with torch.cuda.amp.autocast(enabled=self.amp):
            with torch.inference_mode():
                outputs = self.model(**batch)

        return outputs

    def backward(self, loss: torch.Tensor) -> None:
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
        else:
            loss.backward()

    def clip_grad_norm(self) -> None:
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

    def step(self) -> None:
        # Unscale gradients.
        if self.grad_scaler is not None:
            self.grad_scaler.unscale_(self.optimizer)

        # Clip gradients.
        self.clip_grad_norm()

        # Take optimizer step.
        if self.grad_scaler is not None:
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            self.optimizer.step()

        # Adjust LR schedule.
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def get_model_state(self) -> "OrderedDict[str, torch.Tensor]":
        if self.train_config.is_distributed:
            return self.model.module.state_dict()  # type: ignore[union-attr]
        else:
            return self.model.state_dict()

    def load_model_state(self, state_dict: "OrderedDict[str, torch.Tensor]") -> None:
        if self.train_config.is_distributed:
            self.model.module.load_state_dict(state_dict)  # type: ignore[union-attr]
        else:
            self.model.load_state_dict(state_dict)  # type: ignore[union-attr]

    def save_checkpoint(self, path: Path, client_state: Dict[str, Any]) -> None:
        path.mkdir(exist_ok=True)

        def save_state(state: Dict[str, Any], name: str):
            temp_state_file = tempfile.NamedTemporaryFile(
                "w+b", dir=path, delete=False, suffix=".pt"
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
                    temp_state_file.name, path / f"worker{self.train_config.worker_id}_{name}.pt"
                )
            finally:
                if os.path.exists(temp_state_file.name):
                    os.remove(temp_state_file.name)

        save_state(self.get_model_state(), "model")
        save_state(self.optimizer.state_dict(), "optimizer"),
        if self.lr_scheduler is not None:
            save_state(self.lr_scheduler.state_dict(), "lr_scheduler")
        save_state(client_state, "trainer")

    def load_checkpoint(self, path: Path) -> Dict[str, Any]:
        self.load_model_state(torch.load(path / f"worker{self.train_config.worker_id}_model.pt"))
        self.optimizer.load_state_dict(
            torch.load(path / f"worker{self.train_config.worker_id}_optimizer.pt")
        )
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(
                torch.load(path / f"worker{self.train_config.worker_id}_lr_scheduler.pt")
            )
        return torch.load(path / f"worker{self.train_config.worker_id}_trainer.pt")

    def save_complete_weights_from_checkpoint(
        self, checkpoint_path: Path, weights_path: Path
    ) -> None:
        os.link(checkpoint_path.resolve() / "worker0_model.pt", weights_path)
