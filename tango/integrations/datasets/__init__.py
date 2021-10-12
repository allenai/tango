"""
Components for Tango integration with `🤗 Datasets <https://huggingface.co/docs/datasets/>`_.
"""


from typing import Union

import datasets
from datasets import DatasetDict, Dataset, IterableDatasetDict, IterableDataset

from tango.step import Step


__all__ = ["LoadDataset"]


@Step.register("datasets::load")
class LoadDataset(Step):
    """
    This step loads a `HuggingFace dataset <https://huggingface.co/datasets>`_.

    .. tip::

        Registered as a :class:`~tango.step.Step` under the name "datasets::load".

    Examples
    --------

    .. testsetup::

        from tango import Step

    .. testcode::

        load_step = Step.from_params({
            "type": "datasets::load",
            "path": "lhoestq/test",
        })

    """

    DETERMINISTIC = True
    VERSION = "001"
    CACHEABLE = False  # These are already cached by huggingface.

    def run(self, path: str, **kwargs) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:  # type: ignore
        """
        Loads a HuggingFace dataset.

        ``path`` is the canonical name or path to the dataset. Additional key word arguments
        are passed as-is to :func:`datasets.load_dataset()`.
        """
        return datasets.load_dataset(path, **kwargs)
