from typing import Union, Optional, Dict

import datasets as ds

from tango.integrations.datasets import DatasetsFormat
from tango.step import Step


@Step.register("rte-for-t5")
class RTEForT5Step(Step):
    DETERMINISTIC = True
    CACHEABLE = True
    VERSION = "001"

    FORMAT = DatasetsFormat()

    def run(  # type: ignore
        self,
        data: Union[ds.DatasetDict, ds.Dataset],
        source_prefix: str = "rte",
        premise_prefix: str = "sentence1",
        hypothesis_prefix: str = "sentence2",
        label_prefix: Optional[str] = None,
        num_workers: int = 1,
        seq2seq: bool = True,
    ) -> Union[ds.DatasetDict, ds.Dataset]:
        def filter_no_gold(example, indices):
            if example["label"] == -1:
                return False
            return True

        data = data.filter(filter_no_gold, with_indices=True)
        empty_splits = []
        for split in data.keys():
            if data[split].num_rows <= 0:
                empty_splits.append(split)
        for split in empty_splits:
            del data[split]

        label_map = {0: "entailment", 1: "not_entailment"}

        def _seq2seq_mapper(example):
            if label_prefix is None:
                source_finisher = f" {label_prefix}: "
            else:
                source_finisher = ""
            return {
                "source": (
                    f'{source_prefix} {premise_prefix}: {example["sentence1"]} '
                    f'{hypothesis_prefix}: {example["sentence2"]}{source_finisher}'
                ),
                "target": f'{label_map[example["label"]]}',
            }

        def _causal_mapper(example):
            text = (
                f'{source_prefix} {premise_prefix}: {example["sentence1"]} '
                f'{hypothesis_prefix}: {example["sentence2"]} '
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
