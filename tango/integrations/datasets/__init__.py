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


import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar, Union, overload

from tango.common.aliases import PathOrStr
from tango.common.dataset_dict import DatasetDict, IterableDatasetDict
from tango.common.exceptions import ConfigurationError, IntegrationMissingError
from tango.format import Format
from tango.step import Step

try:
    import datasets as ds
except ModuleNotFoundError:
    raise IntegrationMissingError("datasets")

__all__ = [
    "LoadDataset",
    "LoadStreamingDataset",
    "DatasetsFormat",
    "convert_to_tango_dataset_dict",
    "InterleaveDatasets",
    "ConcatenateDatasets",
    "DatasetRemixStep",
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

    VERSION = "001"

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


@Step.register("datasets::dataset_remix")
class DatasetRemixStep(Step):
    """
    This step can remix splits in a :class:`~datasets.DatasetDict` into new splits.

    .. tip::

        Registered as a :class:`~tango.step.Step` under the name "datasets::dataset_remix".

    Examples
    --------

    .. testcode::
        :hide:

        from tango.common.logging import initialize_logging
        initialize_logging(enable_cli_logs=True)
        import datasets

    .. testcode::

        input = datasets.load_dataset("lhoestq/test")
        new_splits = {
            "all": "train + validation",
            "crossval_train": "train[:1] + validation[1:]",
            "crossval_test": "train[1:] + validation[:1]",
        }
        step = DatasetRemixStep()
        remixed_dataset = step.run(input=input, new_splits=new_splits)

    .. testoutput::
        :hide:
        :options: +ELLIPSIS

        ...

    """

    DETERMINISTIC = True
    CACHEABLE = True
    VERSION = "001"

    def run(  # type: ignore
        self,
        input: ds.DatasetDict,
        new_splits: Dict[str, str],
        keep_old_splits: bool = True,
        shuffle_before: bool = False,
        shuffle_after: bool = False,
        random_seed: int = 1532637578,
    ) -> ds.DatasetDict:
        """
        Remixes and shuffles a dataset. This is done eagerly with native 🤗 Datasets features.

        :param input:
            The input dataset that will be remixed.
        :param new_splits:
            Specifies the new splits that the output dataset should have. Keys are the name of the new
            splits. Values refer to the original splits. You can refer to original splits in the following ways:

            * Mention the original split name to copy it to a new name.
            * Mention the original split name with Python's slicing syntax to select part of the original
              split's instances. For example, ``"train[:1000]"`` selects the first 1000 instances from the
              ``"train"`` split.
            * ``"instances + instances"`` concatenates the instances into one split.

            You can combine these possibilities.
        :param keep_old_splits:
            Whether to keep the splits from the input dataset in addition to the new ones given by
            ``new_splits``.
        :param shuffle_before:
            Whether to shuffle the input splits before creating the new ones.

            If you need shuffled instances and you're not sure the input is properly shuffled, use this.
        :param shuffle_after:
            Whether to shuffle the input splits after creating the new ones.

            If you need shuffled instances and you're slicing or concatenating splits, use this.

            If you want to be on the safe side, shuffle both before and after.
        :param random_seed:
            Random seed, affects shuffling

        :returns:
            Returns a new dataset that is appropriately remixed.
        """

        if shuffle_before:
            input = input.shuffle(random_seed)

        def get_slice(split_name: str) -> ds.Dataset:
            slice_match = re.match(r"(.*)\[(-?[0-9]*:-?[0-9]*)\]", split_name)
            if slice_match is None:
                return input[split_name]
            else:
                split_name = slice_match[1]
                slice_args = [int(a) if len(a) > 0 else None for a in slice_match[2].split(":")]
                slice_indices = range(*slice(*slice_args).indices(len(input[split_name])))
                return input[split_name].select(slice_indices)

        def parse_split_spec(split_spec: str):
            parts = [get_slice(name.strip()) for name in split_spec.split("+")]
            if len(parts) == 1:
                return parts[0]
            else:
                return ds.concatenate_datasets(parts)

        if keep_old_splits:
            result = ds.DatasetDict(input.items())
        else:
            result = ds.DatasetDict()
        result.update(
            {
                new_split_name: parse_split_spec(new_split_spec)
                for new_split_name, new_split_spec in new_splits.items()
            }
        )

        if shuffle_after:
            result = result.shuffle(random_seed)

        return result
