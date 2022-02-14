import datasets

from tango import Step
from tango.integrations.datasets import DatasetsFormat
from tango.integrations.transformers import Tokenizer


# We need a step to tokenize the raw data. The result of this step will be passed
# directly into the "torch::train" step.
@Step.register("tokenize_data")
class TokenizeData(Step):
    DETERMINISTIC = True
    CACHEABLE = True
    FORMAT = DatasetsFormat()

    def run(  # type: ignore[override]
        self,
        dataset: datasets.DatasetDict,
        tokenizer: Tokenizer,
        block_size: int = 1024,
        num_workers: int = 1,
        field_to_tokenize: str = "text",
    ) -> datasets.DatasetDict:
        def tokenize_function(example):
            return tokenizer(example[field_to_tokenize])

        dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=num_workers,
            remove_columns=list(dataset.column_names.values())[0],  # remove all old columns
            desc="Tokenizing dataset",
        )

        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}  # type: ignore
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported
            # it instead of this drop, you can customize this part to your needs.
            if total_length >= block_size:
                total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        dataset = dataset.map(
            group_texts,
            batched=True,
            num_proc=num_workers,
            desc=f"Grouping texts into chunks of {block_size}",
        )

        return dataset
