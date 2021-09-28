"""
Components for Tango integration with `🤗 Datasets <https://huggingface.co/docs/datasets/>`_.
"""


from typing import Union

import datasets
from datasets import DatasetDict, Dataset, IterableDatasetDict, IterableDataset

from tango.step import Step


__all__ = ["HuggingFaceDataset"]


@Step.register("hf_dataset")
class HuggingFaceDataset(Step):
    """
    This steps loads a HuggingFace dataset.

    .. tip::

        Registered as a :class:`~tango.step.Step` under the name ``hf_dataset``.

    """

    DETERMINISTIC = True
    VERSION = "001"
    CACHEABLE = False  # These are already cached by huggingface.

    def run(self, path: str, **kwargs) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:  # type: ignore
        """Reads and returns a huggingface dataset. `dataset_name` is the name of the dataset."""
        return datasets.load_dataset(path, **kwargs)
