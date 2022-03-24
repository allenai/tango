import logging
from typing import Any, Dict, Optional

import datasets as ds
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from tango.integrations.datasets import DatasetsFormat
from tango.integrations.torch import Model
from tango.integrations.torch.util import set_seed_all
from tango.integrations.transformers.tokenizer import Tokenizer
from tango.step import Step

logger = logging.getLogger(__name__)


@Model.register("transformers::finetune-wrapper")
class FinetuneWrapper(Model):
    def __init__(
        self, pretrained_model_name_or_path: str, tokenizer: Optional[Tokenizer] = None, **kwargs
    ):
        super().__init__()
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                pretrained_model_name_or_path, **kwargs
            )
            self.seq2seq = True  # Seq2Seq models don't return their own prefix.
        except ValueError:
            self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path)
            self.seq2seq = False

        if tokenizer:
            # TODO: is this required? This is the only reason why we have tokenizer here.
            self.model.resize_token_embeddings(len(tokenizer))

    def forward(self, *args, **kwargs):
        # TODO: decode and compute other metrics?
        return self.model.forward(*args, **kwargs)


# def _model_for_finetuning(
#     model_name: str,
#     tokenizer: Optional[Tokenizer] = None,
#     max_source_length: Optional[int] = 1024,
#     resize_position_embeddings: Optional[bool] = None,
#     seed: int = 42,
# ):
#     set_seed_all(seed)
#     tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
#
#     try:
#         model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
#         seq2seq_model = True  # Seq2Seq models don't return their own prefix.
#     except ValueError:
#         model = AutoModelForCausalLM.from_pretrained(model_name)
#         seq2seq_model = False
#
#     # TODO: is this required? This is the only reason why we have tokenizer here.
#     model.resize_token_embeddings(len(tokenizer))
#
#     # TODO: MBart specific tokenizer update.
#
#     if model.config.decoder_start_token_id is None:
#         raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
#
#     if (
#         hasattr(model.config, "max_position_embeddings")
#         and model.config.max_position_embeddings < max_source_length
#     ):
#         if resize_position_embeddings is None:
#             logger.warning(
#                 f"Increasing the model's number of position embedding vectors from {model.config.max_position_embeddings} "
#                 f"to {max_source_length}."
#             )
#             model.resize_position_embeddings(max_source_length)
#         elif resize_position_embeddings:
#             model.resize_position_embeddings(max_source_length)
#         else:
#             raise ValueError(
#                 f"`max_source_length` is set to {max_source_length}, but the model only has {model.config.max_position_embeddings}"
#                 f" position encodings. Consider either reducing `max_source_length` to {model.config.max_position_embeddings} or to automatically "
#                 "resize the model's position encodings by setting `resize_position_embeddings`."
#             )
#
#         return model


# @Step.register("get-model-for-finetuning")
# class GetModelForFinetuning(Step):
#     DETERMINISTIC = True
#     CACHEABLE = False
#
#     def run(self, model):


@Step.register("tokenize_text2text")
class TokenizeText2TextData(Step):
    DETERMINISTIC = True
    CACHEABLE = True
    FORMAT = DatasetsFormat()

    def run(  # type: ignore[override]
        self,
        data: ds.DatasetDict,
        tokenizer: Tokenizer,
        num_workers: int = 1,
        source_field: str = "source",
        target_field: str = "target",
        max_source_length: Optional[int] = 1024,
        max_target_length: Optional[int] = 1024,
        pad_to_max_length: bool = False,
        ignore_pad_token_for_loss: bool = True,
    ) -> ds.DatasetDict:

        # Set max_target_length for training.
        max_target_length = max_target_length
        padding = "max_length" if pad_to_max_length else False

        def preprocess_function(examples):
            # remove pairs where at least one record is None
            inputs, targets = [], []
            for i in range(len(examples[source_field])):
                if examples[source_field][i] is not None and examples[target_field][i] is not None:
                    inputs.append(examples[source_field][i])
                    targets.append(examples[target_field][i])

            model_inputs = tokenizer(
                inputs, max_length=max_source_length, padding=padding, truncation=True
            )

            # Setup the tokenizer for targets
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    targets, max_length=max_target_length, padding=padding, truncation=True
                )

            # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
            # padding in the loss.
            if padding == "max_length" and ignore_pad_token_for_loss:
                labels["input_ids"] = [
                    [(l if l != tokenizer.pad_token_id else -100) for l in label]
                    for label in labels["input_ids"]
                ]

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        data = data.map(
            preprocess_function,
            batched=True,
            num_proc=num_workers,
            remove_columns=list(data.column_names.values())[0],  # remove all old columns
            desc="Tokenizing dataset",
        )

        return data
