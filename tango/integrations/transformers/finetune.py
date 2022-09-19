import logging
from os import PathLike
from typing import List, Optional, Union, cast

import datasets as ds
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    PreTrainedModel,
)

from tango.common import Lazy, Params
from tango.format import Format
from tango.integrations.datasets import DatasetsFormat, convert_to_tango_dataset_dict
from tango.integrations.torch import (
    DataCollator,
    DataLoader,
    Model,
    TorchFormat,
    TrainCallback,
    TrainingEngine,
)
from tango.integrations.torch.train import TorchTrainStep
from tango.integrations.transformers import Tokenizer
from tango.step import Step

logger = logging.getLogger(__name__)

SEQ2SEQ = AutoModelForSeq2SeqLM._model_mapping.keys()  # type: ignore
CAUSAL = AutoModelForCausalLM._model_mapping.keys()  # type: ignore


class FinetuneWrapper(PreTrainedModel):
    """
    Wrapper `PreTrainedModel` class that returns either a `Seq2SeqLM` or `CausalLM` model.
    """

    @classmethod
    def from_pretrained(  # type: ignore
        cls,
        pretrained_model_name_or_path: Union[str, PathLike],
        num_tokens: Optional[int] = None,
        **kwargs,
    ) -> PreTrainedModel:
        """
        :param pretrained_model_name_or_path:
            The name of the model to return. Any name that works in the transformers library works here.
        :param num_tokens:
            The number of token embeddings to have.
        """
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path, **kwargs)
        except ValueError:
            model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, **kwargs)

        if num_tokens is not None:
            model.resize_token_embeddings(num_tokens)
        return model


Model.register("transformers::finetune::from_pretrained", constructor="from_pretrained")(
    FinetuneWrapper
)


def _add_special_tokens(tokenizer: Tokenizer) -> None:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.sep_token is None:
        tokenizer.add_special_tokens({"sep_token": "[SEP]"})
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "[EOS]"})


def tokenize_data(
    data: ds.DatasetDict,
    tokenizer: Tokenizer,
    num_workers: int = 1,
    source_field: str = "source",
    target_field: str = "target",
    max_source_length: Optional[int] = 1024,
    max_target_length: Optional[int] = 1024,
    pad_to_max_length: bool = False,
    ignore_pad_token_for_loss: bool = True,
    concat_source_target: bool = False,
) -> ds.DatasetDict:
    """
    Returns a `DatasetDict` with tokenized source and target fields.

    :param data:
        The original dataset dict containing the source and target fields.
    :param tokenizer:
        The tokenizer to use.
    :param num_workers:
        The number of workers to use for processing the data.
    :param source_field:
        The string name of the field containing the source sequence.
    :param target_field:
        The string name of the field containing the target sequence.
    :param max_source_length:
        The maximum number of tokens in the source sequence.
    :param max_target_length:
        The maximum number of tokens in the target sequence.
    :param pad_to_max_length:
        Whether to pad to the maximum length when tokenizing.
    :param ignore_pad_token_for_loss:
        Whether to ignore the padded tokens for calculating loss.
        If set to True, all the pad tokens in the labels are replaced
        by -100, which is ignored by the loss function.
    :param concat_source_target:
        If the downstream model is decoder-only, like "gpt2", the source
        and target sequences need to be concatenated and fed to the model
        together.
    """
    padding = "max_length" if pad_to_max_length else False

    _add_special_tokens(tokenizer)

    def preprocess_function(examples):
        # remove pairs where at least one record is None
        inputs, targets = [], []
        input_lengths = []
        for i in range(len(examples[source_field])):
            if examples[source_field][i] is not None and examples[target_field][i] is not None:
                if not concat_source_target:
                    inputs.append(examples[source_field][i])
                    targets.append(examples[target_field][i])
                else:
                    text = (
                        examples[source_field][i]
                        + tokenizer.sep_token
                        + examples[target_field][i]
                        + tokenizer.eos_token
                    )
                    inputs.append(text)
                    targets.append(text)
                    input_lengths.append(len(examples[source_field][i]))

        model_inputs = tokenizer(
            inputs, max_length=max_source_length, padding=padding, truncation=True
        )

        if not concat_source_target:
            # Setup the tokenizer for targets
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    targets, max_length=max_target_length, padding=padding, truncation=True
                )
        else:
            labels = {"input_ids": []}
            for input_ids in model_inputs["input_ids"]:
                label_start_idx = input_ids.index(tokenizer.sep_token_id)
                label_ids = [-100] * len(input_ids)
                label_ids[label_start_idx + 1 :] = input_ids[label_start_idx + 1 :]
                labels["input_ids"].append(label_ids)

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


@Step.register("transformers::tokenize_text2text")
class TokenizeText2TextData(Step):
    """
    A step that tokenizes data containing source and target sequences.

    .. tip::
        Registered as a :class:`~tango.step.Step` under the name "transformers::tokenize_text2text".
    """

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
        concat_source_target: bool = False,
    ) -> ds.DatasetDict:
        """
        Returns a `DatasetDict` with tokenized source and target fields.

        :param data:
            The original dataset dict containing the source and target fields.
        :param tokenizer:
            The tokenizer to use.
        :param num_workers:
            The number of workers to use for processing the data.
        :param source_field:
            The string name of the field containing the source sequence.
        :param target_field:
            The string name of the field containing the target sequence.
        :param max_source_length:
            The maximum number of tokens in the source sequence.
        :param max_target_length:
            The maximum number of tokens in the target sequence.
        :param pad_to_max_length:
            Whether to pad to the maximum length when tokenizing.
        :param ignore_pad_token_for_loss:
            Whether to ignore the padded tokens for calculating loss.
            If set to True, all the pad tokens in the labels are replaced
            by -100, which is ignored by the loss function.
        :param concat_source_target:
            If the downstream model is decoder-only, like "gpt2", the source
            and target sequences need to be concatenated and fed to the model
            together.

        .. tip::
            If concat_source_target is set to True, we pad all sequences to max
            length here. Otherwise, we leave it to the appropriate
            :class:`~tango.integrations.torch.DataCollator` object.
        """
        return tokenize_data(
            data,
            tokenizer=tokenizer,
            num_workers=num_workers,
            source_field=source_field,
            target_field=target_field,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            pad_to_max_length=pad_to_max_length,
            ignore_pad_token_for_loss=ignore_pad_token_for_loss,
            concat_source_target=concat_source_target,
        )


@Step.register("transformers::finetune")
class FinetuneStep(TorchTrainStep):
    """
    Mostly similar to :class:`~tango.integrations.torch.train.TorchTrainStep` with additional
    preprocessing for data.

    .. tip::

        Registered as a :class:`~tango.step.Step` under the name "transformers::finetune".

    .. important::

        The training loop will use GPU(s) automatically when available, as long as at least
        ``device_count`` CUDA devices are available.

        Distributed data parallel training is activated when the ``device_count`` is greater than 1.

        You can control which CUDA devices to use with the environment variable ``CUDA_VISIBLE_DEVICES``.
        For example, to only use the GPUs with IDs 0 and 1, set ``CUDA_VISIBLE_DEVICES=0,1``
        (and ``device_count`` to 2).

    .. warning::

        During validation, the validation metric (specified by the ``val_metric_name`` parameter)
        is aggregated by simply averaging across validation batches and distributed processes.
        This behavior is usually correct when your validation metric is "loss" or "accuracy",
        for example, but may not be correct for other metrics like "F1".

        If this is not correct for your metric you will need to handle the aggregation
        internally in your model or with a :class:`TrainCallback`
        using the :meth:`TrainCallback.post_val_batch()` method.
        Then set the parameter ``auto_aggregate_val_metric`` to ``False``.

        Note that correctly aggregating your metric during distributed training will
        involve distributed communication.

    """

    DETERMINISTIC = True
    CACHEABLE = True
    FORMAT: Format = TorchFormat()
    SKIP_ID_ARGUMENTS = {"distributed_port", "log_every"}

    def run(  # type: ignore[override]
        self,
        model: Lazy[Model],
        tokenizer: Tokenizer,
        training_engine: Lazy[TrainingEngine],
        dataset_dict: ds.DatasetDict,
        train_dataloader: Lazy[DataLoader],
        *,
        train_split: str = "train",
        validation_split: Optional[str] = None,
        validation_dataloader: Optional[Lazy[DataLoader]] = None,
        source_field: str = "source",
        target_field: str = "target",
        max_source_length: Optional[int] = 1024,
        max_target_length: Optional[int] = 1024,
        seed: int = 42,
        train_steps: Optional[int] = None,
        train_epochs: Optional[int] = None,
        validation_steps: Optional[int] = None,
        grad_accum: int = 1,
        log_every: int = 10,
        checkpoint_every: int = 100,
        validate_every: Optional[int] = None,
        device_count: int = 1,
        distributed_port: int = 54761,
        val_metric_name: str = "loss",
        minimize_val_metric: bool = True,
        auto_aggregate_val_metric: bool = True,
        callbacks: Optional[List[Lazy[TrainCallback]]] = None,
        remove_stale_checkpoints: bool = True,
    ) -> Model:
        """
        Run a basic training loop to train the ``model``.

        :param model:
            The model to train. It should return a ``dict`` that includes the ``loss``
            during training and the ``val_metric_name`` during validation.
        :param tokenizer:
            The tokenizer to use for tokenizing source and target sequences.
        :param training_engine:
            The :class:`TrainingEngine` to use to train the model.
        :param dataset_dict:
            The train and optional validation data.
        :param train_dataloader:
            The data loader that generates training batches. The batches should be :class:`dict`
            objects that will be used as ``kwargs`` for the model's ``forward()`` method.
        :param train_split:
            The name of the data split used for training in the ``dataset_dict``.
            Default is "train".
        :param validation_split:
            Optional name of the validation split in the ``dataset_dict``. Default is ``None``,
            which means no validation.
        :param validation_dataloader:
            An optional data loader for generating validation batches. The batches should be
            :class:`dict` objects. If not specified, but ``validation_split`` is given,
            the validation ``DataLoader`` will be constructed from the same parameters
            as the train ``DataLoader``.
        :param source_field:
            The string name of the field containing the source sequence.
        :param target_field:
            The string name of the field containing the target sequence.
        :param max_source_length:
            The maximum number of tokens in the source sequence.
        :param max_target_length:
            The maximum number of tokens in the target sequence.
        :param seed:
            Used to set the RNG states at the beginning of training.
        :param train_steps:
            The number of steps to train for. If not specified training will
            stop after a complete iteration through the ``train_dataloader``.
        :param train_epochs:
            The number of epochs to train for. You cannot specify ``train_steps`` and ``train_epochs``
            at the same time.
        :param validation_steps:
            The number of steps to validate for. If not specified validation
            will stop after a complete iteration through the ``validation_dataloader``.
        :param grad_accum:
            The number of gradient accumulation steps. Defaults to 1.

            .. note::
                This parameter - in conjuction with the settings of your data loader
                and the number distributed workers -
                determines the *effective batch size* of your training run.

        :param log_every:
            Log every this many steps.
        :param checkpoint_every:
            Save a checkpoint every this many steps.
        :param validate_every:
            Run the validation loop every this many steps.
        :param device_count:
            The number of devices to train on, i.e. the number of distributed data parallel workers.
        :param distributed_port:
            The port of the distributed process group. Default = "54761".
        :param val_metric_name:
            The name of the validation metric, i.e. the key of the metric in the dictionary
            returned by the forward pass of the model. Default is "loss".
        :param minimize_val_metric:
            Whether the validation metric is meant to be minimized (such as the loss).
            Default is ``True``. When using a metric such as accuracy, you should set
            this to ``False``.
        :param auto_aggregate_val_metric:
            If ``True`` (the default), the validation metric will be averaged across
            validation batches and distributed processes. This may not be the correct
            behavior for some metrics (such as F1), in which you should set this to
            ``False`` and handle the aggregation internally in your model
            or with a :class:`TrainCallback` (using :meth:`TrainCallback.post_val_batch()`).
        :param callbacks:
            A list of :class:`TrainCallback`.
        :param remove_stale_checkpoints:
            If ``True`` (the default), stale checkpoints will be removed throughout training so that
            only the latest and best checkpoints are kept.

        :returns:
            The trained model on CPU with the weights from the best checkpoint loaded.

        """
        devices = self._get_devices(device_count)

        is_distributed = False
        if devices and len(devices) > 1:
            is_distributed = True

        # Setup the tokenizer
        _add_special_tokens(tokenizer)

        # Hacky way to deal with resizing the model embeddings.
        model_params_dict = model._params.as_dict()
        if "fairscale" in model_params_dict["type"]:
            model_params_dict["model"]["num_tokens"] = len(tokenizer)  # type: ignore
        else:
            model_params_dict["num_tokens"] = len(tokenizer)  # type: ignore

        model = Lazy(
            model._constructor,
            Params(model_params_dict),
            constructor_extras=model._constructor_extras,
        )

        # Get the config to check in order to check if the model is seq2seq or causal.
        config = AutoConfig.from_pretrained(tokenizer.name_or_path)
        seq2seq: bool = type(config) in SEQ2SEQ

        dataset_dict = tokenize_data(
            dataset_dict,
            tokenizer=tokenizer,
            source_field=source_field,
            target_field=target_field,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            concat_source_target=not seq2seq,
        )

        if is_distributed:
            from torch.utils.data.distributed import DistributedSampler

            sampler = Lazy(DistributedSampler, drop_last=True, shuffle=True)
            train_dataloader = Lazy(
                train_dataloader._constructor,
                train_dataloader._params,
                constructor_extras=train_dataloader._constructor_extras,
                sampler=sampler,
            )

        collate_fn: DataCollator
        collate_fn = cast(DataCollator, DataCollatorForSeq2Seq(tokenizer=tokenizer))

        train_dataloader = Lazy(
            train_dataloader._constructor,
            train_dataloader._params,
            constructor_extras=train_dataloader._constructor_extras,
            collate_fn=collate_fn,
        )

        return self._train(
            model=model,
            training_engine=training_engine,
            dataset_dict=convert_to_tango_dataset_dict(dataset_dict),
            train_dataloader=train_dataloader,
            train_split=train_split,
            validation_split=validation_split,
            validation_dataloader=validation_dataloader,
            seed=seed,
            train_steps=train_steps,
            train_epochs=train_epochs,
            validation_steps=validation_steps,
            grad_accum=grad_accum,
            log_every=log_every,
            checkpoint_every=checkpoint_every,
            validate_every=validate_every,
            devices=devices,
            distributed_port=distributed_port,
            val_metric_name=val_metric_name,
            minimize_val_metric=minimize_val_metric,
            auto_aggregate_val_metric=auto_aggregate_val_metric,
            callbacks=callbacks,
            remove_stale_checkpoints=remove_stale_checkpoints,
        )
