import logging
from typing import Any, List

import flax
import jax
from flax.training.train_state import TrainState

from .model import Model
from .optim import Optimizer


class FlaxTrainState:
    def __init__(
        self,
        model: Model,
        optimizer: Optimizer,
    ):
        self.model = model
        self.optimizer = optimizer
        self.state = (self.create_train_state(),)
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
            apply_fn=self.model.__call__, params=self.model.params, tx=self.optimizer
        )
        return self.state

    def update_state(self, batch, loss_fn):
        # call loss function with batch?
        # labels = batch.pop("labels")
        grad_fn = jax.value_and_grad(loss_fn)
        loss, grad = grad_fn(self.state.params)
        grad = jax.lax.pmean(grad, "batch")
        self.state = self.state.apply_gradients(grads=grad)
        return self.state, loss

    def eval_state(self, batch, logits_fn):
        logits = self.state.apply_fn(**batch, params=self.state.params, train=False)[0]
        labels = batch.pop("labels")
        loss = logits_fn(logits, labels)
        return loss

    def replicate_state(self):
        self.state = flax.jax_utils.replicate(self.devices)
        return self.state

    def unreplicate_state(self):
        self.state = flax.jax_utils.unreplicate(self.state)
        return self.state

    def save_state(self, step: int):
        raise NotImplementedError
