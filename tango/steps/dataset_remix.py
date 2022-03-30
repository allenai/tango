import collections
import random
import re
from typing import Any, Dict, List, Mapping, Sequence

from tango.common.dataset_dict import DatasetDict
from tango.common.sequences import (
    ConcatenatedSequence,
    ShuffledSequence,
    SlicedSequence,
)
from tango.step import Step


@Step.register("dataset_remix")
class DatasetRemixStep(Step[DatasetDict]):
    """
    This step can remix splits in a :class:`~tango.common.dataset_dict.DatasetDict` into new splits.

    .. tip::

        Registered as a :class:`~tango.step.Step` under the name "dataset_remix".

    Examples
    --------

    .. testcode::
        :hide:

        from tango.common.logging import initialize_logging
        initialize_logging(enable_cli_logs=True)

    .. testcode::

        input = DatasetDict({
            "train": list(range(10)),
            "dev": list(range(10, 15)),
        })
        new_splits = {
            "all": "train + dev",
            "crossval_train": "train[0:5] + train[7:]",
            "crossval_test": "train[5:7]",
        }
        remix_step = DatasetRemixStep(input=input, new_splits=new_splits)
        remixed_dataset = remix_step.result()

    .. testoutput::
        :hide:
        :options: +ELLIPSIS

        ...

    """

    DETERMINISTIC = True
    CACHEABLE = False  # This is so fast it's not worth caching.
    VERSION = "001"

    def run(  # type: ignore
        self,
        input: DatasetDict,
        new_splits: Dict[str, str],
        keep_old_splits: bool = True,
        shuffle_before: bool = False,
        shuffle_after: bool = False,
        random_seed: int = 1532637578,
    ) -> DatasetDict:
        """
        Remixes and shuffles a dataset. This is done lazily, so all operations are fast.

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

            If you want to be on the safe side, shuffle both before and after. Shuffling is a cheap operation.
        :param random_seed:
            Random seed, affects shuffling

        :returns:
            Returns a new dataset that is appropriately remixed.
        """
        random.seed(random_seed)

        if shuffle_before:
            input_splits: Mapping[str, Sequence[Any]] = {
                split_name: ShuffledSequence(split_instances)
                for split_name, split_instances in input.splits.items()
            }
        else:
            input_splits = input.splits

        def get_slice(split_name: str) -> Sequence[Any]:
            slice_match = re.match(r"(.*)\[(-?[0-9]*:-?[0-9]*)\]", split_name)
            if slice_match is None:
                return input[split_name]
            else:
                split_name = slice_match[1]
                slice_args = [int(a) if len(a) > 0 else None for a in slice_match[2].split(":")]
                return SlicedSequence(input[split_name], slice(*slice_args))

        def parse_split_spec(split_spec: str):
            parts = [get_slice(name.strip()) for name in split_spec.split("+")]
            if len(parts) == 1:
                return parts[0]
            else:
                return ConcatenatedSequence(*parts)

        if keep_old_splits:
            result = dict(input_splits.items())
        else:
            result = {}
        result.update(
            {
                new_split_name: parse_split_spec(new_split_spec)
                for new_split_name, new_split_spec in new_splits.items()
            }
        )

        if shuffle_after:
            result = {
                split_name: ShuffledSequence(split_instances)
                for split_name, split_instances in result.items()
            }

        return DatasetDict(splits=result, metadata=input.metadata)


@Step.register("dataset_combine")
class DatasetCombineStep(Step[DatasetDict]):
    """
    This step combines multiple :class:`~tango.common.dataset_dict.DatasetDict` s into one.

    .. tip::

        Registered as a :class:`~tango.step.Step` under the name "dataset_combine".

    Examples
    --------

    .. testcode::
        :hide:

        from tango.common.logging import initialize_logging
        initialize_logging(enable_cli_logs=True)

    .. testcode::

        input1 = DatasetDict({
            "train": list(range(10)),
            "dev": list(range(10, 15)),
        })
        input2 = DatasetDict({
            "train": list(range(15, 25)),
            "val": list(range(25, 30)),
        })
        combined = DatasetCombineStep(inputs=[input1, input2])
        combined_dataset = combined.result()

    .. testoutput::
        :hide:
        :options: +ELLIPSIS

        ...

    """

    DETERMINISTIC = True
    CACHEABLE = False  # This is so fast it's not worth caching.
    VERSION = "001"

    def run(  # type: ignore
        self,
        inputs: List[DatasetDict],
        shuffle: bool = False,
        random_seed: int = 1532637578,
    ) -> DatasetDict:
        """
        Combines multiple datasets into one. This is done lazily, so all operations are fast.

        If a split is present in more than one input dataset, the output dataset will have a split that's
        the concatenation of the input splits.

        :param inputs:
            The list of input datasets that will be combined.
        :param shuffle:
            Whether to shuffle the combined datasets. If you don't do this, the new splits will contain first
            all the instances from one dataset, and then all the instances from another dataset.
        :param random_seed:
            Random seed, affects shuffling

        :returns:
            Returns a new dataset that is the combination of the input datasets.
        """

        split_to_datasets: Dict[str, List[Sequence]] = collections.defaultdict(lambda: [])
        for input in inputs:
            for split_name, sequence in input.items():
                split_to_datasets[split_name].append(sequence)
        result: Dict[str, Sequence] = {
            split_name: ConcatenatedSequence(*sequences)
            for split_name, sequences in split_to_datasets.items()
        }

        if shuffle:
            random.seed(random_seed)
            result = {
                split_name: ShuffledSequence(split_instances)
                for split_name, split_instances in result.items()
            }

        return DatasetDict(result, {})
