from typing import Any, Dict, List, Optional

import datasets
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
)
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from tango import Step
from tango.integrations.datasets import DatasetsFormat
from tango.integrations.torch import DataCollator, LRScheduler, Model, Optimizer

# Register the AdamW optimizer from HF as an `Optimizer` so we can use it in the train step.
Optimizer.register("transformers_adamw")(AdamW)


@Model.register("lm-pretrained")
def from_pretrained(
    pretrained_model_name_or_path: str,
    *args,
    fsdp: bool = False,
    fsdp_reshard_after_forward: bool = True,
    fsdp_move_params_to_cpu: bool = False,
    fsdp_move_grads_to_cpu: Optional[bool] = None,
    fsdp_mixed_precision: bool = False,
    activation_checkpointing: bool = False,
    **kwargs,
) -> Model:
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
    _fairscale_wrap_layers(
        model,
        fsdp=fsdp,
        fsdp_reshard_after_forward=fsdp_reshard_after_forward,
        fsdp_move_params_to_cpu=fsdp_move_params_to_cpu,
        fsdp_move_grads_to_cpu=fsdp_move_grads_to_cpu,
        fsdp_mixed_precision=fsdp_mixed_precision,
        activation_checkpointing=activation_checkpointing,
    )
    return model


@Model.register("lm-fresh")
def new_random_from_config(
    pretrained_model_name_or_path: str,
    fsdp: bool = False,
    fsdp_reshard_after_forward: bool = True,
    fsdp_move_params_to_cpu: bool = False,
    fsdp_move_grads_to_cpu: Optional[bool] = None,
    fsdp_mixed_precision: bool = False,
    activation_checkpointing: bool = False,
) -> Model:
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
    model = AutoModelForCausalLM.from_config(config)  # type: ignore
    _fairscale_wrap_layers(
        model,
        fsdp=fsdp,
        fsdp_reshard_after_forward=fsdp_reshard_after_forward,
        fsdp_move_params_to_cpu=fsdp_move_params_to_cpu,
        fsdp_move_grads_to_cpu=fsdp_move_grads_to_cpu,
        fsdp_mixed_precision=fsdp_mixed_precision,
        activation_checkpointing=activation_checkpointing,
    )
    return model


def _fairscale_wrap_layers(
    model,
    fsdp: bool = False,
    fsdp_reshard_after_forward: bool = True,
    fsdp_move_params_to_cpu: bool = False,
    fsdp_move_grads_to_cpu: Optional[bool] = None,
    fsdp_mixed_precision: bool = False,
    activation_checkpointing: bool = False,
) -> None:
    if activation_checkpointing:
        from fairscale.nn.checkpoint import checkpoint_wrapper

        for block_idx in range(len(model.transformer.h)):
            model.transformer.h[block_idx] = checkpoint_wrapper(
                model.transformer.h[block_idx], offload_to_cpu=True
            )

    if fsdp and torch.distributed.is_initialized():
        from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
        from fairscale.nn.wrap import enable_wrap, wrap

        with enable_wrap(
            wrapper_cls=FSDP,
            reshard_after_forward=fsdp_reshard_after_forward,
            move_params_to_cpu=fsdp_move_params_to_cpu,
            move_grads_to_cpu=fsdp_move_grads_to_cpu,
            mixed_precision=fsdp_mixed_precision,
        ):
            for block_idx in range(len(model.transformer.h)):
                model.transformer.h[block_idx] = wrap(model.transformer.h[block_idx])

    #  def load_final_state_dict(self, state_dict: "OrderedDict[str, torch.Tensor]"):
    #      """
    #      Due to weight tying, `lm_head.weight` might be missing from the state dictionary.
    #      """
    #      missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
    #      if missing_keys and set(missing_keys) != {"lm_head.weight"}:
    #          missing_keys.remove("lm_head.weight")
    #          raise RuntimeError(f"Error loading state dict, missing keys: {missing_keys}")
    #      elif unexpected_keys:
    #          raise RuntimeError(f"Error loading state dict, unexpected keys: {unexpected_keys}")


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
    CACHEABLE = True
    FORMAT = DatasetsFormat()

    def run(  # type: ignore[override]
        self,
        dataset: datasets.DatasetDict,
        pretrained_model_name: str,
        block_size: int = 1024,
        num_workers: int = 1,
    ) -> datasets.DatasetDict:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

        def tokenize_function(example):
            return tokenizer(example["text"])

        dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=num_workers,
            remove_columns=["text"],
            desc="Tokenizing dataset",
            cache_file_names={
                "train": f"/tmp/wikitext2-train-{pretrained_model_name.replace('/', '-')}-tokenized.cache",
                "test": f"/tmp/wikitext2-test-{pretrained_model_name.replace('/', '-')}-tokenized.cache",
                "validation": f"/tmp/wikitext2-dev-{pretrained_model_name.replace('/', '-')}-tokenized.cache",
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
                "train": f"/tmp/wikitext2-train-{pretrained_model_name.replace('/', '-')}-chunked.cache",
                "test": f"/tmp/wikitext2-test-{pretrained_model_name.replace('/', '-')}-chunked.cache",
                "validation": f"/tmp/wikitext2-dev-{pretrained_model_name.replace('/', '-')}-chunked.cache",
            },
        )

        return dataset
