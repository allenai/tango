from typing import Any, Generic, TypeVar

import jax.numpy as jnp
import jax.random

from flax.training.common_utils import shard

from tango.common.registrable import Registrable

T = TypeVar("T")


class DataLoader(Generic[T], Registrable):
    """
    A :class:`~tango.common.Registrable` which  will take in dataset and act as a dataloader for Flax models.
    """

    def __init__(self, dataset: Any, batch_size: int, drop_last: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last


@DataLoader.register("flax::numpy_dataloader")
class NumpyLoader(DataLoader):
    """
    The class will take in dataset in form of numpy arrays and convert them into batches of jax
    device arrays.
    """

    def __call__(self, rng):
        steps_per_epoch = len(self.dataset) // self.batch_size
        perms = jax.random.permutation(rng, len(self.dataset))
        perms = perms[: steps_per_epoch * self.batch_size]  # Skip incomplete batch.
        perms = perms.reshape((steps_per_epoch, self.batch_size))
        for perm in perms:
            batch = self.dataset[perm]
            batch = jnp.array(batch)
            batch = shard(batch)
            yield batch


@DataLoader.register("flax::dataset_dataloader")
class DatasetLoader(DataLoader):
    """
    The class will take in Datasets object and covert it into batches of jax device arrays.
    """

    def __call__(self, rng):
        steps_per_epoch = len(self.dataset) // self.batch_size
        perms = jax.random.permutation(rng, len(self.dataset))
        perms = perms[: steps_per_epoch * self.batch_size]  # Skip incomplete batch.
        perms = perms.reshape((steps_per_epoch, self.batch_size))
        for perm in perms:
            batch = self.dataset[perm]
            batch = {k: jnp.array(v) for k, v in batch.items()}
            batch = shard(batch)
            yield batch
