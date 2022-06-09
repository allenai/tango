# TODO: Define a Flax Data collator. It should accept input of type numpy and convert them into jax arrays in batches -
#  we want this to be a generator function (similar to a data collator)

from typing import Generic, TypeVar, Any

import jax.numpy as jnp
import jax.random
import numpy as np

from tango.common.registrable import Registrable
from .util import GetPRNGkey

T = TypeVar("T")


class DataLoader(Generic[T], Registrable):
    """
    A :class:`~tango.common.Registrable` which  will take in dataset and act as a dataloader for Flax models.
    """


@DataLoader.register("flax::numpy_dataloader")
class NumpyLoader(DataLoader):
    """
    The class will take in dataset in form of numpy arrays and convert them into jax
    device arrays.
    """

    def __init__(
            self,
            dataset: np.array,
            batch_size: int,
            drop_last: bool = True
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __call__(self):
        rng = GetPRNGkey()
        shuffled_idx = jax.random.permutation(rng, self.dataset.shape[0])
        shuffled = self.dataset[[shuffled_idx]]
        for i in range(len(shuffled) // self.batch_size):
            batch = shuffled[i * self.batch_size: (i + 1) * self.batch_size]
            batch = jnp.array(batch)
            yield batch
