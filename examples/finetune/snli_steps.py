from typing import Union

import datasets as ds

from tango.integrations.datasets import DatasetsFormat
from tango.step import Step


@Step.register("subset-data")
class SubsetData(Step):
    """
    Creates a subset of the data; mostly to be used for testing/debugging.
    """

    DETERMINISTIC = True
    CACHEABLE = True
    VERSION = "001"

    FORMAT = DatasetsFormat()

    def run(  # type: ignore
        self,
        data: Union[ds.DatasetDict, ds.Dataset],
        max_samples: int = 5,
    ) -> Union[ds.DatasetDict, ds.Dataset]:
        """
        Returns a copy of the `data` with number of samples limited to `max_samples` for
        each split.

        :param data:
            The dataset or dataset dict object.
        :param max_samples:
            The maximum number of samples to return per split.
        """

        # Unlike `ds.Dataset.select`, this works on both `ds.Dataset` and `ds.DatasetDict`.
        def filter_fn(example, indices):
            return indices < max_samples

        return data.filter(filter_fn, with_indices=True)


@Step.register("snli-text2text")
class SnliText2Text(Step):
    """
    Converts the snli dataset to a text-to-text format.

    Examples
    --------

    original_instance = {
        "premise": "Two cats are sitting on a wall.",
        "hypothesis": "The cats are chasing a mouse.",
        "label": 2  # contradiction
    }

    returned_instance = {
        "source": "nli premise: Two cats are sitting on a wall. hypothesis: The cats are chasing a mouse. label: "
        "target": "contradiction"
    }

    """

    DETERMINISTIC = True
    CACHEABLE = True
    VERSION = "001"

    FORMAT = DatasetsFormat()

    def run(  # type: ignore
        self,
        data: Union[ds.DatasetDict, ds.Dataset],
        source_prefix: str = "nli",
        premise_prefix: str = "premise",
        hypothesis_prefix: str = "hypothesis",
        label_prefix: str = "label",
        num_workers: int = 1,
    ) -> Union[ds.DatasetDict, ds.Dataset]:
        """
        :param data:
            The snli `Dataset` or `DatasetDict` object.
        :param source_prefix:
            The str to add before the start of the source sequence.
        :param premise_prefix:
            The str to add before the start of the `premise` in the source sequence.
        :param hypothesis_prefix:
            The str to add before the start of the `hypothesis` in the source sequence.
        :param label_prefix:
            The str to add as the prompt for the label.
        :param num_workers:
            The number of workers to use for processing the data.
        """

        def filter_no_gold(example, indices):
            if example["label"] == -1:
                return False
            return True

        data = data.filter(filter_no_gold, with_indices=True)

        label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}

        def _mapper(example):
            return {
                "source": (
                    f'{source_prefix} {premise_prefix}: {example["premise"]} '
                    f'{hypothesis_prefix}: {example["hypothesis"]} {label_prefix}: '
                ),
                "target": f'{label_map[example["label"]]}',
            }

        if isinstance(data, ds.Dataset):
            old_cols = data.column_names
        else:
            old_cols = list(data.column_names.values())[0]

        dataset = data.map(
            _mapper,
            batched=False,
            num_proc=num_workers,
            remove_columns=old_cols,  # remove all old columns
            desc="Converting data to text-to-text format",
        )

        return dataset
