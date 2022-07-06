import jax
import jax.numpy as jnp
import optax
import tensorflow_datasets as tfds
from flax import linen as nn

from tango.integrations.flax import FlaxEvalWrapper, FlaxTrainWrapper, Model
from tango.step import Step

# TODO: Fix the example and move it to examples/


@Step.register("load_mnist_data")
class LoadMNISTData(Step):
    DETERMINISTIC = True
    CACHEABLE = True

    def run(self):
        ds_builder = tfds.builder("mnist")
        ds_builder.download_and_prepare()
        train_ds = tfds.as_numpy(ds_builder.as_dataset(split="train", batch_size=-1))
        train_ds["x"] = train_ds["image"] / 255.0
        train_ds["labels"] = train_ds["label"]
        train_ds["num_rows"] = len(train_ds["x"])
        test_ds = tfds.as_numpy(ds_builder.as_dataset(split="test", batch_size=-1))
        test_ds["x"] = test_ds["image"] / 255.0
        test_ds["labels"] = test_ds["label"]
        test_ds["num_rows"] = len(test_ds["x"])
        dataset = {"train": train_ds, "test": test_ds}
        return dataset


@Model.register("mnist")
class MNIST(Model):
    """
    A simple CNN model
    """

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x


@FlaxTrainWrapper.register("mnist_train_wrapper")
class TrainWrapper(FlaxTrainWrapper):
    def __init__(self):
        self.model = MNIST()

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

    def loss_fn(self, params, batch, state, dropout_rng):
        """
        Compute loss and metrics during train.
        """

        def compute_loss(logits, labels):
            labels_onehot = jax.nn.one_hot(labels, num_classes=10)
            loss = optax.softmax_cross_entropy(logits=logits, labels=labels_onehot).mean()
            return loss

        labels = batch["labels"]
        logits = self.model.apply({"params": params}, batch["x"])
        loss = compute_loss(logits, labels)
        return loss, logits

    def eval_fn(self, batch, state, model):
        """
        Compute loss and metrics during eval.
        """
        logits = self.model.apply({"params": state.params}, batch["x"])
        return logits


@FlaxEvalWrapper.register("mnist_eval_wrapper")
class EvalWrapper(FlaxEvalWrapper):
    def __init__(self):
        self.model = MNIST()

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

    def eval_step(self, state, batch):
        logits = self.model.apply({"params": state.params}, batch["x"])
        metrics = self.compute_metrics(logits=logits, labels=batch["label"])
        return logits, metrics
