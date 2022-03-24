from typing import Union

import datasets as ds

from tango.integrations.datasets import DatasetsFormat
from tango.step import Step


@Step.register("subset-data")
class SubsetData(Step):
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
        Unlike `ds.Dataset.select`, this works on both `ds.Dataset` and `ds.DatasetDict`.
        """

        def filter_fn(example, indices):
            return indices < max_samples

        return data.filter(filter_fn, with_indices=True)


@Step.register("snli-text2text")
class SnliText2Text(Step):
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
        seq2seq: bool = True,
    ) -> Union[ds.DatasetDict, ds.Dataset]:
        def filter_no_gold(example, indices):
            if example["label"] == -1:
                return False
            return True

        data = data.filter(filter_no_gold, with_indices=True)

        label_map = {0: "entails", 1: "neutral", 2: "contradiction"}

        def _seq2seq_mapper(example):
            return {
                "source": (
                    f'{source_prefix} {premise_prefix}: {example["premise"]} '
                    f'{hypothesis_prefix}: {example["hypothesis"]} {label_prefix}: '
                ),
                "target": f'{label_map[example["label"]]}',
            }

        def _causal_mapper(example):
            text = (
                f'{source_prefix} {premise_prefix}: {example["premise"]} '
                f'{hypothesis_prefix}: {example["hypothesis"]} '
                f'{label_prefix}: {label_map[example["label"]]}'
            )
            return {"source": text, "target": text}

        if isinstance(data, ds.Dataset):
            old_cols = data.column_names
        else:
            old_cols = list(data.column_names.values())[0]

        _mapper = _seq2seq_mapper if seq2seq else _causal_mapper

        dataset = data.map(
            _mapper,
            batched=False,
            num_proc=num_workers,
            remove_columns=old_cols,  # remove all old columns
            desc="Converting data to seq2seq format",
        )

        return dataset
