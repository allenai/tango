import logging
from typing import Generic, TypeVar

import jax.random
import numpy as np
from flax.training.common_utils import shard

from tango.common.dataset_dict import DatasetDictBase
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
        dataset: DatasetDictBase,
        batch_size: int = 8,
        drop_last: bool = True,
        shuffle: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.dataset_size = self._get_size()

        self.logger = logging.getLogger(FlaxDataLoader.__name__)

    def _get_size(self):
        size = self.dataset["num_rows"] if type(self.dataset) is dict else self.dataset.num_rows
        return size

    def __call__(self, rng: jax.random.PRNGKeyArray, do_distributed: bool):
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
