from typing import Any, Dict, List

import datasets
from transformers import GPT2LMHeadModel, GPT2Tokenizer, default_data_collator
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from tango import Step
from tango.common import DatasetDict
from tango.integrations.datasets import convert_to_tango_dataset_dict
from tango.integrations.torch import DataCollator, LRScheduler, Model, Optimizer

# Register the AdamW optimizer from HF as an `Optimizer` so we can use it in the train step.
Optimizer.register("transformers_adamw")(AdamW)

# Similarly for our model.
Model.register("gpt2", constructor="from_pretrained")(GPT2LMHeadModel)


# We also want to use `get_linear_schedule_with_warmup()` from HF, but we need a class
# to work with, so we just create this dummy class with a classmethod that will call
# `get_linear_schedule_with_warmup()`.
@LRScheduler.register("linear_with_warmup", constructor="linear_with_warmup")
class TransformersLambdaLR(LRScheduler):
    @classmethod
    def linear_with_warmup(cls, optimizer: Optimizer, **kwargs) -> LRScheduler:
        return get_linear_schedule_with_warmup(optimizer, **kwargs)


# And we also want to use the `default_data_collator()` function from HF as a `DataCollator`,
# so we create simple wrapper class around that function and register it.
@DataCollator.register("transformers_default")
class TransformerDefaultCollator(DataCollator[Any]):
    def __call__(self, items: List[Any]) -> Dict[str, Any]:
        return default_data_collator(items)


# Lastly, we need a step to tokenize the raw data. The result of this step will be passed
# directly into the "torch::train" step.
@Step.register("tokenize_data")
class TokenizeData(Step):
    DETERMINISTIC = True
    CACHEABLE = False

    def run(  # type: ignore[override]
        self,
        dataset: datasets.DatasetDict,
        pretrained_model_name: str,
        block_size: int = 1024,
        num_workers: int = 1,
    ) -> DatasetDict:
        tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name)

        def tokenize_function(example):
            return tokenizer(example["text"])

        dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=num_workers,
            remove_columns=["text"],
            desc="Tokenizing dataset",
            cache_file_names={
                "train": f"/tmp/wikitext2-train-{pretrained_model_name.replace('/', '-')}-tokenized",
                "test": f"/tmp/wikitext2-test-{pretrained_model_name.replace('/', '-')}-tokenized",
                "validation": f"/tmp/wikitext2-dev-{pretrained_model_name.replace('/', '-')}-tokenized",
            },
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
            cache_file_names={
                "train": f"/tmp/wikitext2-train-{pretrained_model_name.replace('/', '-')}-chunked",
                "test": f"/tmp/wikitext2-test-{pretrained_model_name.replace('/', '-')}-chunked",
                "validation": f"/tmp/wikitext2-dev-{pretrained_model_name.replace('/', '-')}-chunked",
            },
        )

        # It's important for caching any steps that use this dataset as input
        # to convert it to a native Tango DatasetDict.
        return convert_to_tango_dataset_dict(dataset)
