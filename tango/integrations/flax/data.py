import logging
from typing import Generic, TypeVar

import jax.random
import numpy as np
from datasets import Dataset
from flax.training.common_utils import shard

from tango.common.registrable import Registrable

T = TypeVar("T")


class DataLoader(Generic[T], Registrable):
    """
    A :class:`~tango.common.Registrable` version of a ``Flax DataLoader``.
    ``Flax DataLoader`` accepts Dataset object. The class yields a numpy batch.
    """


@DataLoader.register("flax::dataloader")
class FlaxDataLoader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 8,
        drop_last: bool = True,
        shuffle: bool = True,
    ):
        self.dataset = dataset
        self.dataset_size = dataset.num_rows
        self.batch_size = batch_size
        self.drop_last = drop_last
        if not drop_last:
            raise NotImplementedError(
                "With Jax you have to drop the last incomplete batch, because the batch size is compiled into the "
                "model."
            )
        self.shuffle = shuffle

        self.logger = logging.getLogger(FlaxDataLoader.__name__)

    def __call__(self, rng: jax._src.random.KeyArrayLike, do_distributed: bool):
        steps_per_epoch = self.dataset_size // self.batch_size

        if self.shuffle:
            perms = jax.random.permutation(rng, self.dataset_size)
            perms = np.asarray(perms)  # using jax arrays for indexing is a bottleneck on TPUs.
        else:
            perms = np.arange(self.dataset_size)

        self.logger.info("Skipping last incomplete batch")
        perms = perms[: steps_per_epoch * self.batch_size]  # Skip incomplete batch.
        perms = perms.reshape((steps_per_epoch, self.batch_size))

        for perm in perms:
            batch = self.dataset[perm]
            if do_distributed:
                batch = shard(batch)
            yield batch
