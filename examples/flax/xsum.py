from typing import Optional

import jax
import jax.numpy as jnp
import nltk
import numpy as np
import optax
from datasets import load_metric
from flax.training.common_utils import onehot
from transformers import AutoConfig, AutoTokenizer, FlaxAutoModelForSeq2SeqLM

from tango.integrations.flax import FlaxWrapper
from tango.integrations.flax.train_callback import TrainCallback
from tango.step import Step

"""
XSum Summarization with facebook/bart-base
"""


@Step.register("tokenize_data")
class PreProcessing(Step):
    DETERMINISTIC = False

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


@FlaxWrapper.register("xsum_wrapper")  # type: ignore
class TransformerWrapper(FlaxWrapper):
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

    def train_loss(self, params, state, batch, dropout_rng, labels):
        logits = state.apply_fn(**batch, params=params, dropout_rng=dropout_rng, train=True)[0]
        loss = self.loss_helper(logits, labels, batch)
        return loss

    def val_metrics(self, batch, logits, labels):
        loss = self.loss_helper(logits, labels, batch)
        metrics = {"loss": loss}
        return metrics

    def eval_metrics(self, batch, logits, labels):
        loss = self.loss_helper(logits, labels, batch)
        metrics = {"loss": loss}
        return metrics


@TrainCallback.register("flax::generate_step")
class GenerateCallback(TrainCallback):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def generate_step(self, params, batch):
        self.model.params = params
        gen_kwargs = {"max_length": 64, "num_beams": self.model.config.num_beams}
        output_ids = self.model.generate(
            batch["input_ids"], attention_mask=batch["attention_mask"], **gen_kwargs
        )
        return output_ids.sequences

    def pre_train_loop(self) -> None:
        if len(jax.devices()) > 1:
            self.p_generate_step = jax.pmap(self.generate_step, axis_name="batch")

    def pre_val_loop(self, step: int, val_step: int, state) -> None:
        self.state = state
        self.eval_preds = []
        self.eval_labels = []

    def pre_val_batch(self, step: int, val_step: int, epoch: int, val_batch) -> None:
        labels = val_batch["labels"]
        if len(jax.devices()) > 1:
            print(len(val_batch))
            generated_ids = self.p_generate_step(self.state.params, val_batch)
        else:
            generated_ids = self.generate_step(self.state.params, val_batch)
        self.eval_preds.extend(jax.device_get(generated_ids.reshape(-1, 64)))
        self.eval_labels.extend(jax.device_get(labels.reshape(-1, labels.shape[-1])))

    def postprocess_text(self, preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(self, preds, labels):
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = self.postprocess_text(decoded_preds, decoded_labels)
        metric = load_metric("rouge")
        result = metric.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )
        # Extract a few results from ROUGE
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    def post_val_loop(
        self, step: int, epoch: int, val_metric: Optional[float], best_val_metric: Optional[float]
    ) -> None:
        rouge_metrics = self.compute_metrics(self.eval_preds, self.eval_labels)
        rouge_desc = " ".join([f"Eval {key}: {value} |" for key, value in rouge_metrics.items()])
        print(rouge_desc)
