from typing import Any, Generic, TypeVar

import datasets
import jax.numpy as jnp
import jax.random
import numpy as np
from flax.training.common_utils import shard

from tango.common.registrable import Registrable

T = TypeVar("T")


class DataLoader(Generic[T], Registrable):
    """
    A :class:`~tango.common.Registrable` which  will take in dataset and act as a dataloader for Flax models.
    """


@DataLoader.register("flax::dataloader")
class FlaxDataLoader(DataLoader):
    """
    The class will take in dataset in form of numpy arrays and convert them into batches of jax
    device arrays.
    """

    def __init__(
        self, dataset: Any, batch_size: int = 1, drop_last: bool = True, shuffle: bool = True
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

    def __call__(self, rng: jax.random.PRNGKeyArray):
        steps_per_epoch = len(self.dataset) // self.batch_size

        if self.shuffle:
            perms = jax.random.permutation(rng, len(self.dataset))
        else:
            perms = jax.numpy.arange(len(self.dataset))

        perms = perms[: steps_per_epoch * self.batch_size]  # Skip incomplete batch.
        perms = perms.reshape((steps_per_epoch, self.batch_size))

        for perm in perms:
            if isinstance(self.dataset, np.ndarray):
                batch = self.dataset[[perm]]
                batch = jnp.array(batch)
            elif isinstance(self.dataset, datasets.Dataset):
                batch = self.dataset[perm]
                batch = {k: jnp.array(v) for k, v in batch.items()}
            else:
                raise ValueError("Type for FlaxLoader should be numpy.ndarray or Datasets")
            batch = shard(batch)
            yield batch
