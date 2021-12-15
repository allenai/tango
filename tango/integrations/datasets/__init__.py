"""
.. important::
    To use this integration you should install ``tango`` with the "datasets" extra
    (e.g. ``pip install tango[datasets]``) or just install the ``datasets`` library after the fact
    (e.g. ``pip install datasets``).

Components for Tango integration with `🤗 Datasets <https://huggingface.co/docs/datasets/>`_.

Example: loading and combining
------------------------------

Here's an example config that uses the built-in steps from this integration to load,
concatenate, and interleave datasets from HuggingFace:

.. literalinclude:: ../../../../test_fixtures/integrations/datasets/config.json

You could run this with:

.. code-block::

    tango run config.json

"""


from pathlib import Path
from typing import Any, List, Optional, TypeVar, Union, overload

import datasets as ds

from tango.common.aliases import PathOrStr
from tango.common.dataset_dict import DatasetDict, DatasetDictBase, IterableDatasetDict
from tango.common.exceptions import ConfigurationError
from tango.format import Format
from tango.step import Step

__all__ = [
    "LoadDataset",
    "LoadStreamingDataset",
    "DatasetsFormat",
    "convert_to_tango_dataset_dict",
    "InterleaveDatasets",
    "ConcatenateDatasets",
]


@overload
def convert_to_tango_dataset_dict(hf_dataset_dict: ds.DatasetDict) -> DatasetDict:
    ...


@overload
def convert_to_tango_dataset_dict(hf_dataset_dict: ds.IterableDatasetDict) -> IterableDatasetDict:
    ...


def convert_to_tango_dataset_dict(hf_dataset_dict):
    """
    A helper function that can be used to convert a HuggingFace :class:`~datasets.DatasetDict`
    or :class:`~datasets.IterableDatasetDict` into a native Tango
    :class:`~tango.common.DatasetDict` or :class:`~tango.common.IterableDatasetDict`.

    This is important to do when your dataset dict is input to another step for caching
    reasons.
    """
    if isinstance(hf_dataset_dict, ds.IterableDatasetDict):
        return IterableDatasetDict(splits=hf_dataset_dict)
    else:
        return DatasetDict(splits=hf_dataset_dict)


T = Union[ds.Dataset, ds.DatasetDict]


@Format.register("datasets")
class DatasetsFormat(Format[T]):
    """
    This format writes a :class:`datasets.Dataset` or :class:`datasets.DatasetDict` to disk
    using :meth:`datasets.Dataset.save_to_disk()`.

    It is the default :class:`~tango.format.Format` for the :class:`LoadDataset` step.
    """

    VERSION = 1

    def write(self, artifact: T, dir: PathOrStr):
        dataset_path = Path(dir) / "data"
        artifact.save_to_disk(str(dataset_path))

    def read(self, dir: PathOrStr) -> T:
        dataset_path = Path(dir) / "data"
        return ds.load_from_disk(str(dataset_path))


@Step.register("datasets::load")
class LoadDataset(Step):
    """
    This step loads a `HuggingFace dataset <https://huggingface.co/datasets>`_.

    .. tip::

        Registered as a :class:`~tango.step.Step` under the name "datasets::load".

    .. important::

        If you are loading an :class:`~datasets.IterableDataset` or :class:`~datasets.IterableDatasetDict`
        you need to use the :class:`LoadStreamingDataset` step instead.

    """

    DETERMINISTIC = True
    VERSION = "001"
    CACHEABLE = True
    # Even though HuggingFace datasets has its own caching mechanism, it can still be worth caching
    # this step with tango's mechanism since some datasets take a really long time to query from HuggingFace
    # ("bigscience/P3", for example). Tango's caching mechanism circumvents that issue.
    FORMAT = DatasetsFormat()

    def run(self, path: str, **kwargs) -> Union[ds.DatasetDict, ds.Dataset]:  # type: ignore
        """
        Load the HuggingFace dataset specified by ``path``.

        ``path`` is the canonical name or path to the dataset. Additional key word arguments
        are passed as-is to :func:`datasets.load_dataset()`.
        """
        dataset = ds.load_dataset(path, **kwargs)
        if not isinstance(dataset, (ds.Dataset, ds.DatasetDict)):
            raise ConfigurationError(
                f"{self.__class__.__name__} can only be used with non-streaming datasets. "
                f"For streaming datasets, use the 'LoadStreamingDataset' ('datasets::load_streaming') step instead."
            )
        return dataset


@Step.register("datasets::load_streaming")
class LoadStreamingDataset(Step):
    """
    This step loads an iterable/streaming `HuggingFace dataset <https://huggingface.co/datasets>`_.

    .. tip::

        Registered as a :class:`~tango.step.Step` under the name "datasets::load_streaming".

    """

    DETERMINISTIC = True
    VERSION = "001"
    CACHEABLE = (
        False  # can't be cached with `DatasetsFormat`, and might be really inefficient anyway.
    )

    def run(  # type: ignore
        self, path: str, **kwargs
    ) -> Union[ds.IterableDatasetDict, ds.IterableDataset]:
        """
        Load the HuggingFace streaming dataset specified by ``path``.

        ``path`` is the canonical name or path to the dataset. Additional key word arguments
        are passed as-is to :func:`datasets.load_dataset()`.
        """
        dataset = ds.load_dataset(path, **kwargs)
        if not isinstance(dataset, (ds.IterableDataset, ds.IterableDatasetDict)):
            raise ConfigurationError(
                f"{self.__class__.__name__} can only be used with streaming datasets. "
                f"For non-streaming datasets, use the 'LoadDataset' ('datasets::load') step instead."
            )
        return dataset


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
    ) -> DatasetType:
        """
        Interleave the list of datasets.
        """
        return ds.interleave_datasets(datasets, probabilities=probabilities, seed=seed)


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
