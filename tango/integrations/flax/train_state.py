import logging
from typing import Any, List, Callable

import flax
import jax
from flax import struct
from flax.training import train_state

from .model import Model
from .optim import Optimizer, LRScheduler
from .loss import LossFunction

class TrainState(train_state.TrainState):
    logits_fn: Callable = struct.field(pytree_node=False)
    loss_fn: Callable = struct.field(pytree_node=False)

class FlaxTrainState:
    def __init__(
        self,
        model: Model,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        loss_fn: LossFunction
    ):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler,
        self.loss_fn = loss_fn
        self.state = self.create_train_state()
        self.devices = self._get_devices()
        self.logger = logging.getLogger(FlaxTrainState.__name__)

    def _get_devices(self) -> List[Any]:
        device_type = jax.default_backend()
        self.devices = jax.devices()
        device_count = len(self.devices)
        self.logger.info("Training on %d %s", device_count, device_type)
        return self.devices

    def create_train_state(self):
        self.state = TrainState.create(
            apply_fn=self.model.__call__,
            params=self.model.params,
            tx=self.optimizer,
            logits_fn=lambda logits: logits.argmax(-1),
            loss_fn=self.loss_fn
        )
        return self.state

    def update_state(self, batch, dropout_rng):
        labels = batch.pop("labels")

        def loss_fn():
            logits = self.state.apply_fn(**batch, params=self.state.params, dropout_rng=dropout_rng, train=True)[0]
            loss = self.state.loss_fn(logits, labels)
            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grad = grad_fn(self.state.params)
        grad = jax.lax.pmean(grad, "batch")
        self.state = self.state.apply_gradients(grads=grad)
        return self.state, loss

    def eval_state(self, batch):
        logits = self.state.apply_fn(**batch, params=self.state.params, train=False)[0]
        labels = batch.pop("labels")
        loss = self.state.logits_fn(logits, labels)
        return loss

    def replicate_state(self):
        self.state = flax.jax_utils.replicate(self.devices)
        return self.state

    def unreplicate_state(self):
        self.state = flax.jax_utils.unreplicate(self.state)
        return self.state

    def save_state(self, step: int):
        raise NotImplementedError
