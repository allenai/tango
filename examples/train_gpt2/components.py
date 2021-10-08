"""
Components for fine-tuning a standard sized GPT-2 LM on WikiText2 or a similar dataset.
"""

import typing as t

from tango.common.dataset_dict import DatasetDict
from tango.step import Step
from tango.integrations.torch import Model, DataCollator, Optimizer, LRScheduler
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    default_data_collator,
)
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import datasets


Optimizer.register("transformers_adamw")(AdamW)


@Model.register("gpt2", constructor="from_pretrained")
class GPT2Model(GPT2LMHeadModel, Model):
    pass


class TransformersLambdaLR(LRScheduler):
    @classmethod
    def linear_with_warmup(cls, optimizer: Optimizer, **kwargs) -> LRScheduler:
        return get_linear_schedule_with_warmup(optimizer, **kwargs)


LRScheduler.register("linear_with_warmup", constructor="linear_with_warmup")(TransformersLambdaLR)


@DataCollator.register("transformers_default")
class TransformerDefaultCollator(DataCollator[t.Any]):
    def __call__(self, items: t.List[t.Any]) -> t.Dict[str, t.Any]:
        return default_data_collator(items)


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
                "train": f"/tmp/wikitext2-train-{pretrained_model_name}-tokenized",
                "test": f"/tmp/wikitext2-test-{pretrained_model_name}-tokenized",
                "validation": f"/tmp/wikitext2-dev-{pretrained_model_name}-tokenized",
            },
        )

        group_texts = get_group_texts_function(block_size)

        dataset = dataset.map(
            group_texts,
            batched=True,
            num_proc=num_workers,
            desc=f"Grouping texts into chunks of {block_size}",
            cache_file_names={
                "train": f"/tmp/wikitext2-train-{pretrained_model_name}-chunked",
                "test": f"/tmp/wikitext2-test-{pretrained_model_name}-chunked",
                "validation": f"/tmp/wikitext2-dev-{pretrained_model_name}-chunked",
            },
        )

        return t.cast(DatasetDict, dataset)


def get_group_texts_function(block_size: int):
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}  # type: ignore
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    return group_texts
