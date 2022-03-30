import logging
from typing import Optional

import datasets as ds
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, PreTrainedModel

from tango.integrations.datasets import DatasetsFormat
from tango.integrations.torch import Model, TorchFormat
from tango.integrations.transformers.tokenizer import Tokenizer
from tango.step import Step

logger = logging.getLogger(__name__)


class FinetuneWrapper(PreTrainedModel):
    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, tokenizer: Optional[Tokenizer] = None, **kwargs
    ) -> PreTrainedModel:
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path, **kwargs)
        except ValueError:
            model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, **kwargs)

        if tokenizer:
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            if tokenizer.sep_token is None:
                tokenizer.add_special_tokens({"sep_token": "[SEP]"})
            if tokenizer.eos_token is None:
                tokenizer.add_special_tokens({"eos_token": "[EOS]"})
            # TODO: is this required? This is the only reason why we have tokenizer here.
            model.resize_token_embeddings(len(tokenizer))  # type: ignore
        return model


Model.register("transformers::finetune::from_pretrained", constructor="from_pretrained")(
    FinetuneWrapper
)


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
        seq2seq: bool = True,
    ) -> ds.DatasetDict:

        if not seq2seq:
            pad_to_max_length = True  # TODO: address this.
        padding = "max_length" if pad_to_max_length else False

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        if tokenizer.sep_token is None:
            tokenizer.add_special_tokens({"sep_token": "[SEP]"})
        if tokenizer.eos_token is None:
            tokenizer.add_special_tokens({"eos_token": "[EOS]"})

        def preprocess_function(examples):
            # remove pairs where at least one record is None
            inputs, targets = [], []
            input_lengths = []
            for i in range(len(examples[source_field])):
                if examples[source_field][i] is not None and examples[target_field][i] is not None:
                    if seq2seq:
                        inputs.append(examples[source_field][i])
                        targets.append(examples[target_field][i])
                    else:
                        text = examples[source_field][i] + " " + examples[target_field][i]
                        inputs.append(text)
                        targets.append(text)
                        input_lengths.append(len(examples[source_field][i]))

            model_inputs = tokenizer(
                inputs, max_length=max_source_length, padding=padding, truncation=True
            )

            # Setup the tokenizer for targets
            with tokenizer.as_target_tokenizer():
                # TODO: do something so the loss isn't counted.
                labels = tokenizer(
                    targets, max_length=max_target_length, padding=padding, truncation=True
                )

            # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100
            # when we want to ignore padding in the loss.
            if padding == "max_length" and ignore_pad_token_for_loss:
                labels["input_ids"] = [
                    [(lb if lb != tokenizer.pad_token_id else -100) for lb in label]
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

# from tango.integrations.torch.train import *  # TODO: fix
#
# @Step.register("transformers::finetune")
# class FinetuneStep(Step):
#     """
#     Mostly similar to :class:`~tango.integrations.torch.train.TorchTrainStep` with additional
#     preprocessing for data.
#
#     .. tip::
#
#         Registered as a :class:`~tango.step.Step` under the name "transformers::finetune".
#
#     .. important::
#
#         The training loop will use GPU(s) automatically when available, as long as at least
#         ``device_count`` CUDA devices are available.
#
#         Distributed data parallel training is activated when the ``device_count`` is greater than 1.
#
#         You can control which CUDA devices to use with the environment variable ``CUDA_VISIBLE_DEVICES``.
#         For example, to only use the GPUs with IDs 0 and 1, set ``CUDA_VISIBLE_DEVICES=0,1``
#         (and ``device_count`` to 2).
#
#     .. warning::
#
#         During validation, the validation metric (specified by the ``val_metric_name`` parameter)
#         is aggregated by simply averaging across validation batches and distributed processes.
#         This behavior is usually correct when your validation metric is "loss" or "accuracy",
#         for example, but may not be correct for other metrics like "F1".
#
#         If this is not correct for your metric you will need to handle the aggregation
#         internally in your model or with a :class:`TrainCallback`
#         using the :meth:`TrainCallback.post_val_batch()` method.
#         Then set the parameter ``auto_aggregate_val_metric`` to ``False``.
#
#         Note that correctly aggregating your metric during distributed training will
#         involve distributed communication.
#
#     """
#
#     DETERMINISTIC = True
#     CACHEABLE = True
#     FORMAT: Format = TorchFormat()
#     SKIP_ID_ARGUMENTS = {"distributed_port", "log_every"}
#
#     def run(  # type: ignore[override]
#         self,
#         model: Lazy[Model],
#         training_engine: Lazy[TrainingEngine],
#         dataset_dict: DatasetDictBase,
#         train_dataloader: Lazy[DataLoader],
#         *,
#         train_split: str = "train",
#         validation_split: Optional[str] = None,
#         validation_dataloader: Optional[Lazy[DataLoader]] = None,
#         seed: int = 42,
#         train_steps: Optional[int] = None,
#         train_epochs: Optional[int] = None,
#         validation_steps: Optional[int] = None,
#         grad_accum: int = 1,
#         log_every: int = 10,
#         checkpoint_every: int = 100,
#         validate_every: Optional[int] = None,
#         device_count: int = 1,
#         distributed_port: int = 54761,
#         val_metric_name: str = "loss",
#         minimize_val_metric: bool = True,
#         auto_aggregate_val_metric: bool = True,
#         callbacks: Optional[List[Lazy[TrainCallback]]] = None,
#         remove_stale_checkpoints: bool = True,
#     ) -> Model:
#         """
#         Run a basic training loop to train the ``model``.
#
#         :param model:
#             The model to train. It should return a ``dict`` that includes the ``loss``
#             during training and the ``val_metric_name`` during validation.
#         :param training_engine:
#             The :class:`TrainingEngine` to use to train the model.
#         :param dataset_dict:
#             The train and optional validation data.
#         :param train_dataloader:
#             The data loader that generates training batches. The batches should be :class:`dict`
#             objects that will be used as ``kwargs`` for the model's ``forward()`` method.
#         :param train_split:
#             The name of the data split used for training in the ``dataset_dict``.
#             Default is "train".
#         :param validation_split:
#             Optional name of the validation split in the ``dataset_dict``. Default is ``None``,
#             which means no validation.
#         :param validation_dataloader:
#             An optional data loader for generating validation batches. The batches should be
#             :class:`dict` objects. If not specified, but ``validation_split`` is given,
#             the validation ``DataLoader`` will be constructed from the same parameters
#             as the train ``DataLoader``.
#         :param seed:
#             Used to set the RNG states at the beginning of training.
#         :param train_steps:
#             The number of steps to train for. If not specified training will
#             stop after a complete iteration through the ``train_dataloader``.
#         :param train_epochs:
#             The number of epochs to train for. You cannot specify ``train_steps`` and ``train_epochs``
#             at the same time.
#         :param validation_steps:
#             The number of steps to validate for. If not specified validation
#             will stop after a complete iteration through the ``validation_dataloader``.
#         :param grad_accum:
#             The number of gradient accumulation steps. Defaults to 1.
#
#             .. note::
#                 This parameter - in conjuction with the settings of your data loader
#                 and the number distributed workers -
#                 determines the *effective batch size* of your training run.
#
#         :param log_every:
#             Log every this many steps.
#         :param checkpoint_every:
#             Save a checkpoint every this many steps.
#         :param validate_every:
#             Run the validation loop every this many steps.
#         :param device_count:
#             The number of devices to train on, i.e. the number of distributed data parallel workers.
#         :param distributed_port:
#             The port of the distributed process group. Default = "54761".
#         :param val_metric_name:
#             The name of the validation metric, i.e. the key of the metric in the dictionary
#             returned by the forward pass of the model. Default is "loss".
#         :param minimize_val_metric:
#             Whether the validation metric is meant to be minimized (such as the loss).
#             Default is ``True``. When using a metric such as accuracy, you should set
#             this to ``False``.
#         :param auto_aggregate_val_metric:
#             If ``True`` (the default), the validation metric will be averaged across
#             validation batches and distributed processes. This may not be the correct
#             behavior for some metrics (such as F1), in which you should set this to
#             ``False`` and handle the aggregation internally in your model
#             or with a :class:`TrainCallback` (using :meth:`TrainCallback.post_val_batch()`).
#         :param callbacks:
#             A list of :class:`TrainCallback`.
#         :param remove_stale_checkpoints:
#             If ``True`` (the default), stale checkpoints will be removed throughout training so that
#             only the latest and best checkpoints are kept.
#
#         :returns:
#             The trained model on CPU with the weights from the best checkpoint loaded.
#
#         """
#         # Validate device(s).
#         if device_count <= 0:
#             raise ConfigurationError("Invalid value for 'device_count'. Must be at least 1.")
#         devices: List[int]
#         if torch.cuda.is_available() and torch.cuda.device_count() >= device_count:
#             devices = list(range(device_count))
#             self.logger.info("Training on %d GPU%s", device_count, "s" if device_count > 1 else "")
#         else:
#             devices = [-1] * device_count
#             self.logger.info(
#                 "Training on CPU with %d worker%s", device_count, "s" if device_count > 1 else ""
#             )
#
#         if validate_every is not None and validation_split is None:
#             raise ConfigurationError(
#                 "You have set a validation interval, but no validation split. "
#                 "That's probably unintentional."
#             )
#
#         is_distributed = False
#         num_workers = 1
#         if devices and len(devices) > 1:
#             is_distributed = True
#             num_workers = len(devices)
#
#         if (train_steps is not None) == (train_epochs is not None):
#             raise ConfigurationError(
#                 "One of 'train_steps' or 'train_epochs' needs to be specified, but not both."
#             )
#
#         # Tokenize data
#
#         # dataset dict
#
#         # end tokenization
#
#         config = TrainConfig(
#             self.unique_id,
#             self.work_dir,
#             train_split=train_split,
#             validation_split=validation_split,
#             seed=seed,
#             train_steps=train_steps,
#             train_epochs=train_epochs,
#             grad_accum=grad_accum,
#             log_every=log_every,
#             checkpoint_every=checkpoint_every,
#             validate_every=validate_every,
#             validation_steps=validation_steps,
#             is_distributed=is_distributed,
#             devices=devices,
#             distributed_port=distributed_port,
#             val_metric_name=val_metric_name,
#             minimize_val_metric=minimize_val_metric,
#             auto_aggregate_val_metric=auto_aggregate_val_metric,
#             remove_stale_checkpoints=remove_stale_checkpoints,
#             world_size=num_workers,
#         )
#
#         final_model: Model
#         if is_distributed:
#             import torch.multiprocessing as mp
#
#             mp.spawn(
#                 _train,
#                 args=(
#                     config,
#                     model,
#                     training_engine,
#                     dataset_dict,
#                     train_dataloader,
#                     validation_dataloader,
#                     callbacks,
#                     get_extra_imported_modules(),
#                 ),
#                 nprocs=num_workers,
#             )
#             self.logger.info("Constructing final model")
#             final_model = model.construct()
#         else:
#             final_model = _train(  # type: ignore[assignment]
#                 0,
#                 config,
#                 model,
#                 training_engine,
#                 dataset_dict,
#                 train_dataloader,
#                 validation_dataloader=validation_dataloader,
#                 callbacks=callbacks,
#             )
#             assert final_model is not None
#             final_model = final_model.cpu()
#
#         # Load best checkpoint before returning model.
#         if config.final_weights_path.is_file():
#             self.logger.info(
#                 f"Loading best weights from {str(config.final_weights_path.resolve())}"
#             )
#             state = torch.load(config.final_weights_path, map_location="cpu")
#             # We use `strict=False` because there might be missing keys due to weight tying.
#             final_model.load_state_dict(state, strict=False)
#
#         return final_model