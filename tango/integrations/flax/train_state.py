from abc import abstractmethod
from typing import Dict, Optional, Tuple

import flax
import jax
import jax.numpy as jnp
import optax
from flax.serialization import from_state_dict, to_state_dict
from flax.training.train_state import TrainState

from .model import Model
from .optim import LRScheduler, Optimizer
from .util import get_PRNGkey


class FlaxTrainState:
    def __init__(
        self,
        model: Model,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        do_distributed: bool,
        shape: Optional[list] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = (lr_scheduler,)
        self.do_distributed = do_distributed
        self.state = self.create_train_state(shape)

    # user defined
    def compute_metrics(self, logits, labels):

        def cross_entropy_loss(logits, labels):
            labels_onehot = jax.nn.one_hot(labels, num_classes=10)
            return optax.softmax_cross_entropy(logits=logits, labels=labels_onehot).mean()

        loss = cross_entropy_loss(logits=logits, labels=labels)
        accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
        metrics = {
            "loss": loss,
            "accuracy": accuracy,
        }
        return metrics

    # user defined
    def loss_fn(self, params, batch, dropout_rng):
        """
        Compute loss and metrics during train.
        """

        def compute_loss(logits, labels):
            labels_onehot = jax.nn.one_hot(labels, num_classes=10)
            loss = optax.softmax_cross_entropy(logits=logits, labels=labels_onehot).mean()
            return loss

        labels = batch["label"]
        logits = self.model.apply({"params": params}, batch["image"])
        loss = compute_loss(logits, labels)
        return loss, logits

    # user defined
    def eval_fn(self, params, batch) -> Tuple[jnp.ndarray, dict]:
        """
        Compute loss and metrics during eval.
        """
        logits = self.model.apply({"params": params}, batch["image"])
        return logits

    # class functions
    def create_train_state(self, shape):
        try:
            params = self.model.params
        except:
            x = jnp.ones(shape)
            params = self.model.init(get_PRNGkey(), x)["params"]
        self.state = TrainState.create(
            apply_fn=self.model.__call__, params=params, tx=self.optimizer
        )
        return self.state

    def update_state(self, batch, dropout_rng):
        grad_fn = jax.value_and_grad(self.loss_fn, has_aux=True)
        (_, logits), grad = grad_fn(self.state.params, batch, dropout_rng)
        if self.do_distributed:
            grad = jax.lax.pmean(grad, "batch")
        self.state = self.state.apply_gradients(grads=grad)
        metrics = self.compute_metrics(logits=logits, labels=batch["label"])
        if self.do_distributed:
            metrics = jax.lax.pmean(metrics, axis_name="batch")
        return metrics

    def val_state(self, batch) -> Dict:
        logits = self.eval_fn(self.model.params, batch)
        metrics = self.compute_metrics(logits=logits, labels=batch["label"])
        if self.do_distributed:
            metrics = jax.lax.pmean(metrics, axis_name="batch")
        return metrics

    def replicate_state(self):
        self.state = flax.jax_utils.replicate(self.state)
        return self.state

    def unreplicate_state(self):
        self.state = flax.jax_utils.unreplicate(self.state)
        return self.state

    def load_state(self):
        """
        Loads the state from a file.
        """
        state = {}  # read file and load dict
        from_state_dict(self.state, state)
