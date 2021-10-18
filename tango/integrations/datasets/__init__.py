"""
Components for Tango integration with `🤗 Datasets <https://huggingface.co/docs/datasets/>`_.

.. important::
    To use this integration you should install ``tango`` with the "datasets" extra
    (e.g. ``pip install tango[datasets]``) or just install the ``datasets`` library after the fact
    (e.g. ``pip install datasets``).

Example: loading and combining
------------------------------

Here's an example config that uses the built-in steps from this integration to load,
concatenate, and interleave datasets from HuggingFace:

.. literalinclude:: ../../../../test_fixtures/integrations/datasets/config.json

You could run this with:

.. code-block::

    tango run config.json

"""


from typing import Union, TypeVar, List, Optional, Any

import datasets as ds

from tango.step import Step
from tango.common.dataset_dict import DatasetDict


__all__ = [
    "LoadDataset",
    "convert_to_tango_dataset_dict",
    "InterleaveDatasets",
    "ConcatenateDatasets",
]


def convert_to_tango_dataset_dict(hf_dataset_dict: ds.DatasetDict) -> DatasetDict:
    """
    A helper function that can be used to convert a HuggingFace :class:`~datasets.DatasetDict`
    into a native Tango :class:`~tango.common.dataset_dict.DatasetDict`.

    This is important to do when your dataset dict is input to another step for caching
    reasons.
    """
    fingerprint = ""
    for key, dataset in sorted(hf_dataset_dict.items(), key=lambda x: x[0]):
        fingerprint += f"{key}-{dataset._fingerprint}-"
    return DatasetDict(splits=hf_dataset_dict, fingerprint=fingerprint)


@Step.register("datasets::load")
class LoadDataset(Step):
    """
    This step loads a `HuggingFace dataset <https://huggingface.co/datasets>`_.

    .. tip::

        Registered as a :class:`~tango.step.Step` under the name "datasets::load".

    """

    DETERMINISTIC = True
    VERSION = "001"
    CACHEABLE = False  # These are already cached by huggingface.

    def run(  # type: ignore
        self, path: str, **kwargs
    ) -> Union[ds.DatasetDict, ds.Dataset, ds.IterableDatasetDict, ds.IterableDataset]:
        """
        Load the HuggingFace dataset specified by ``path``.

        ``path`` is the canonical name or path to the dataset. Additional key word arguments
        are passed as-is to :func:`datasets.load_dataset()`.
        """
        return ds.load_dataset(path, **kwargs)


DatasetType = TypeVar("DatasetType", ds.Dataset, ds.IterableDataset)


@Step.register("datasets::interleave")
class InterleaveDatasets(Step):
    """
    This steps interleaves multiple datasets using :func:`~datasets.interleave_datasets()`.

    .. tip::

        Registered as a :class:`~tango.step.Step` under the name "datasets::interleave".

    """

    DETERMINISTIC = True
    VERSION = "001"
    CACHEABLE = False  # Not worth caching

    def run(  # type: ignore[override]
        self,
        datasets: List[DatasetType],
        probabilities: Optional[List[float]] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> DatasetType:
        """
        Interleave the list of datasets.
        """
        return ds.interleave_datasets(datasets, probabilities=probabilities, seed=seed, **kwargs)


@Step.register("datasets::concatenate")
class ConcatenateDatasets(Step):
    """
    This step concatenates multiple datasets using :func:`~datasets.concatenate_datasets()`.

    .. tip::

        Registered as a :class:`~tango.step.Step` under the name "datasets::concatenate".

    """

    DETERMINISTIC = True
    VERSION = "001"
    CACHEABLE = False  # Not worth caching

    def run(  # type: ignore[override]
        self,
        datasets: List[ds.Dataset],
        info: Optional[Any] = None,
        split: Optional[Any] = None,
        axis: int = 0,
    ) -> ds.Dataset:
        """
        Concatenate the list of datasets.
        """
        return ds.concatenate_datasets(datasets, info=info, split=split, axis=axis)
