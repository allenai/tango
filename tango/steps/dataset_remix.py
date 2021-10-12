import random
import re
from typing import Mapping, Any, Sequence, Dict

from tango.step import Step
from tango.common.dataset_dict import DatasetDict
from tango.common.sequences import ShuffledSequence, SlicedSequence, ConcatenatedSequence


@Step.register("dataset_remix")
class DatasetRemixStep(Step):
    """
    This step can remix splits in a :class:`~tango.common.dataset_dict.DatasetDict` into new splits.

    .. tip::

        Registered as a :class:`~tango.step.Step` under the name "dataset_remix".

    Examples
    --------

    .. testsetup::

        from tango.common.dataset_dict import DatasetDict
        from tango.steps import DatasetRemixStep

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
        remix = DatasetRemixStep("remix")
        remix.run(input, new_splits)

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
        Remix the dataset.
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
