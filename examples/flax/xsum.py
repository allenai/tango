import jax.numpy as jnp
import numpy as np
import optax
from flax.training.common_utils import onehot
from transformers import AutoConfig, AutoTokenizer, FlaxAutoModelForSeq2SeqLM

from tango.integrations.flax import FlaxEvalWrapper, FlaxTrainWrapper
from tango.step import Step

"""
XSum Summarization with facebook/bart-base
"""


@Step.register("tokenize_data")
class PreProcessing(Step):
    def run(self, dataset):
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
        model = FlaxAutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")
        model_module = __import__(model.__module__, fromlist=["shift_tokens_tight"])
        shift_tokens_right_fn = getattr(model_module, "shift_tokens_right")
        config = AutoConfig.from_pretrained("facebook/bart-base")

        MAX_SOURCE_LENGTH = 512
        MAX_TGT_LENGTH = 64

        def preprocess_function(examples):
            inputs = examples["document"]
            targets = examples["summary"]
            inputs = [inp for inp in inputs]
            model_inputs = tokenizer(
                inputs,
                max_length=MAX_SOURCE_LENGTH,
                padding="max_length",
                truncation=True,
                return_tensors="np",
            )

            # Setup the tokenizer for targets
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    targets,
                    max_length=MAX_TGT_LENGTH,
                    padding="max_length",
                    truncation=True,
                    return_tensors="np",
                )

            model_inputs["labels"] = labels["input_ids"]
            decoder_input_ids = shift_tokens_right_fn(
                labels["input_ids"], config.pad_token_id, config.decoder_start_token_id
            )
            model_inputs["decoder_input_ids"] = np.asarray(decoder_input_ids)

            # We need decoder_attention_mask so we can ignore pad tokens from loss
            model_inputs["decoder_attention_mask"] = labels["attention_mask"]

            return model_inputs

        column_names = dataset["train"].column_names

        dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=column_names,
            desc="Running tokenizer on dataset",
        )
        return dataset


@FlaxTrainWrapper.register("xsum_train_wrapper")
class TransformerTrainWrapper(FlaxTrainWrapper):
    def compute_metrics(self, state, batch, labels):
        # return empty dict if no other metrics to compute
        return {}

    def loss_helper(self, logits, labels, batch):
        label_smoothing_factor = 0
        padding_mask = batch["decoder_attention_mask"]
        vocab_size = logits.shape[-1]
        confidence = 1.0 - label_smoothing_factor
        low_confidence = (1.0 - confidence) / (vocab_size - 1)
        normalizing_constant = -(
            confidence * jnp.log(confidence)
            + (vocab_size - 1) * low_confidence * jnp.log(low_confidence + 1e-20)
        )
        soft_labels = onehot(labels, vocab_size, on_value=confidence, off_value=low_confidence)

        loss = optax.softmax_cross_entropy(logits, soft_labels)
        loss = loss - normalizing_constant

        # ignore padded tokens from loss
        loss = loss * padding_mask
        loss = loss.sum() / padding_mask.sum()
        return loss

    def train_loss(self, params, state, batch, labels, dropout_rng):
        logits = state.apply_fn(**batch, params=state.params, dropout_rng=dropout_rng, train=True)[
            0
        ]
        loss = self.loss_helper(logits, labels, batch)
        return loss

    def val_loss(self, batch, logits, labels):
        loss = self.loss_helper(logits, labels, batch)
        return loss


@FlaxEvalWrapper.register("xsum_eval_wrapper")
class TransformerEvalWrapper(FlaxEvalWrapper):
    def loss_helper(self, logits, labels, batch):
        label_smoothing_factor = 0
        padding_mask = batch["decoder_attention_mask"]
        vocab_size = logits.shape[-1]
        confidence = 1.0 - label_smoothing_factor
        low_confidence = (1.0 - confidence) / (vocab_size - 1)
        normalizing_constant = -(
            confidence * jnp.log(confidence)
            + (vocab_size - 1) * low_confidence * jnp.log(low_confidence + 1e-20)
        )
        soft_labels = onehot(labels, vocab_size, on_value=confidence, off_value=low_confidence)

        loss = optax.softmax_cross_entropy(logits, soft_labels)
        loss = loss - normalizing_constant

        # ignore padded tokens from loss
        loss = loss * padding_mask
        loss = loss.sum() / padding_mask.sum()
        return loss

    def eval_metrics(self, batch, logits, labels):
        loss = self.loss_helper(logits, labels, batch)
        metrics = {"loss": loss}
        return metrics
