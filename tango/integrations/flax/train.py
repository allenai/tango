import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional

import jax
import jax.numpy as jnp
from flax import jax_utils
from flax.training import checkpoints
from flax.training.train_state import TrainState

from tango.common.dataset_dict import DatasetDictBase
from tango.common.exceptions import ConfigurationError
from tango.common.lazy import Lazy
from tango.common.tqdm import Tqdm
from tango.format import Format
from tango.step import Step
from tango.workspace import Workspace

from .data import FlaxDataLoader
from .format import FlaxFormat
from .model import Model
from .optim import LRScheduler, Optimizer
from .train_callback import TrainCallback
from .train_config import TrainConfig
from .util import get_multiple_keys, get_PRNGkey
from .wrapper import FlaxWrapper

PyTree = Any


@Step.register("flax::train")
class FlaxTrainStep(Step):
    """
    A Flax training step that supports distributed training with configurable dataloaders, callbacks,
    optimizer.

    .. tip::
        Registered as a :class:`~tango.step.Step` under the name "flax::train".

    .. important::
        To train on GPUs and TPUs, installation of jax[cuda] or jax[tpu] is required. Follow the
        instructions here: https://github.com/google/jax to set up jax for GPUs and TPUs.
        Note: CUDA and cuDNN installation is required to run jax on NVidia GPUs. It is recommended to
        install cuDNN in your conda environment using: ``conda install -c anaconda cudnn``.

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

        Jax pre-allocates 90% of GPU memory. If you run into out-of-memory (OOM) issues, please refer
        to this: https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html.

    """

    DETERMINISTIC = True
    CACHEABLE = True
    FORMAT: Format = FlaxFormat()
    SKIP_ID_ARGUMENTS = {"log_every"}
    METADATA = {"artifact_kind": "model"}

    def run(  # type: ignore[override]
        self,
        model: Model,
        dataset: DatasetDictBase,
        optimizer: Lazy[Optimizer],
        train_dataloader: Lazy[FlaxDataLoader],
        *,
        wrapper: FlaxWrapper,
        seed: int = 42,
        keep_checkpoints: int = 5,
        lr_scheduler: Optional[Lazy[LRScheduler]] = None,
        train_split: Optional[str] = None,
        validation_dataloader: Optional[Lazy[FlaxDataLoader]] = None,
        validation_split: Optional[str] = None,
        train_steps: Optional[int] = None,
        train_epoch: Optional[int] = None,
        validation_steps: Optional[int] = None,
        log_every: int = 10,
        checkpoint_every: int = 100,
        validate_every: Optional[int] = None,
        val_metric_name: str = "loss",
        minimize_val_metric: bool = True,
        auto_aggregate_val_metric: bool = True,
        callbacks: Optional[List[Lazy[TrainCallback]]] = None,
        remove_stale_checkpoints: bool = True,
    ) -> PyTree:
        """
        Run a basic training loop to train the ``model``.

        :param model:
            The flax model to train. It should define ``__call__()``. Defining ``setup()`` is Optional.
        :param dataset:
            The train and optional validation dataset.
        :param optimizer:
            The name of the optax Optimizer to use for training.
        :param train_dataloader:
            The dataloader object that generates training batches.
        :param wrapper:
            A Wrapper class that defines ``loss_fn()``, ``eval_fn()`` and ``compute_metrics()``
        :param seed:
            Used to set the PRNG state. By default, ``seed=42``
        :param keep_checkpoints:
            An integer which denotes how many previous checkpoints should be stored while training.
            By default, ``keep_checkpoints=5``
        :param lr_scheduler:
            The name of the learning rate scheduler.
        :param train_split:
            The name of the data split used for training in the ``dataset_dict``.
            Default is "train".
        :param validation_dataloader:
            An optional data loader for generating validation batches. The batches should be
            :class:`dict` objects. If not specified, but ``validation_split`` is given,
            the validation ``DataLoader`` will be constructed from the same parameters
            as the train ``DataLoader``.
        :param validation_split:
            Optional name of the validation split in the ``dataset_dict``. Default is ``None``,
            which means no validation.
        :param train_steps:
            The number of steps to train for. If not specified training will
            stop after a complete iteration through the ``train_dataloader``.
        :param train_epoch:
            The number of epochs to train for. You cannot specify ``train_steps`` and ``train_epochs``
            at the same time.
        :param validation_steps:
            The number of steps to validate for. If not specified validation
            will stop after a complete iteration through the ``validation_dataloader``.
        :param log_every:
            Log every this many steps.
        :param checkpoint_every:
            Save a checkpoint every this many steps.
        :param validate_every:
            Run the validation loop every this many steps.
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
            A list of :class: `TrainCallback`.
        :param remove_stale_checkpoints:
            If ``True`` (the default), stale checkpoints will be removed throughout training so that
            only the latest and best checkpoints are kept.

        :returns:
            The trained model with the last checkpoint loaded.
        """

        return self._train(
            dataset=dataset,
            model=model,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            train_wrapper=wrapper,
            seed=seed,
            keep_checkpoints=keep_checkpoints,
            lr_scheduler=lr_scheduler,
            train_split=train_split,
            validation_split=validation_split,
            validation_dataloader=validation_dataloader,
            train_steps=train_steps,
            train_epochs=train_epoch,
            validation_steps=validation_steps,
            log_every=log_every,
            checkpoint_every=checkpoint_every,
            validate_every=validate_every,
            val_metric_name=val_metric_name,
            minimize_val_metric=minimize_val_metric,
            auto_aggregate_val_metric=auto_aggregate_val_metric,
            callbacks=callbacks,
            remove_stale_checkpoints=remove_stale_checkpoints,
        )

    def _train(
        self,
        model: Model,
        optimizer: Lazy[Optimizer],
        dataset: DatasetDictBase,
        train_dataloader: Lazy[FlaxDataLoader],
        *,
        train_wrapper: FlaxWrapper,
        seed: int = 42,
        keep_checkpoints: int = 5,
        lr_scheduler: Optional[Lazy[LRScheduler]],
        train_split: Optional[str] = "train",
        validation_split: Optional[str] = None,
        validation_dataloader: Optional[Lazy[FlaxDataLoader]] = None,
        train_steps: Optional[int] = None,
        train_epochs: Optional[int] = None,
        validation_steps: Optional[int] = None,
        log_every: int = 10,
        checkpoint_every: int = 100,
        validate_every: Optional[int] = None,
        val_metric_name: str = "loss",
        minimize_val_metric: bool = True,
        auto_aggregate_val_metric: bool = True,
        callbacks: Optional[List[Lazy[TrainCallback]]] = None,
        remove_stale_checkpoints: bool = True,
    ) -> PyTree:

        if validate_every is not None and validation_split is None:
            raise ConfigurationError(
                "You have set a validation interval, but no validation split. "
                "That's probably unintentional."
            )

        if (train_steps is not None) and (train_epochs is not None):
            raise ConfigurationError(
                "One of 'train_steps' or 'train_epochs' needs to be specified, but not both."
            )

        if isinstance(dataset, DatasetDictBase) and train_split is None:
            raise ConfigurationError("Specify the train split for Datasets object.")

        config = TrainConfig(
            self.unique_id,
            self.work_dir,
            step_name=self.name,
            train_split=train_split,
            validation_split=validation_split,
            seed=seed,
            train_steps=train_steps,
            train_epochs=train_epochs,
            log_every=log_every,
            checkpoint_every=checkpoint_every,
            validate_every=validate_every,
            validation_steps=validation_steps,
            val_metric_name=val_metric_name,
            minimize_val_metric=minimize_val_metric,
            auto_aggregate_val_metric=auto_aggregate_val_metric,
            remove_stale_checkpoints=remove_stale_checkpoints,
        )

        optimizer = self._construct_optimizer(optimizer)

        lr_scheduler_: Optional[LRScheduler] = None
        if lr_scheduler is not None:
            lr_scheduler_ = self._construct_lr_scheduler(lr_scheduler)
        lr_scheduler = lr_scheduler_

        final_model: Model

        final_model = self.train_helper(
            self.workspace,
            config,
            model,
            optimizer,
            keep_checkpoints,
            lr_scheduler,
            train_wrapper,
            dataset,
            train_dataloader,
            validation_dataloader,
            callbacks,
        )
        assert final_model is not None

        return final_model

    def train_helper(
        self,
        workspace: Workspace,
        config: TrainConfig,
        model: Model,
        optimizer: Optimizer,
        keep_checkpoints: int,
        lr_scheduler: Optional[LRScheduler],
        train_wrapper: FlaxWrapper,
        dataset: DatasetDictBase,
        train_dataloader: Lazy[FlaxDataLoader],
        validation_dataloader: Optional[Lazy[FlaxDataLoader]] = None,
        callbacks: Optional[List[Lazy[TrainCallback]]] = None,
    ) -> PyTree:

        logger = logging.getLogger(FlaxTrainStep.__name__)

        # construct data loaders
        validation_dataloader_: Optional[FlaxDataLoader] = None
        if config.validation_split is not None:
            validation_dataset = dataset[config.validation_split]
            validation_dataset.set_format("numpy")
            if validation_dataloader is not None:
                validation_dataloader_ = validation_dataloader.construct(dataset=validation_dataset)
            else:
                validation_dataloader_ = train_dataloader.construct(dataset=validation_dataset)

        validation_dataloader = validation_dataloader_

        train_dataset = dataset
        if config.train_split is not None:
            train_dataset = dataset[config.train_split]
            train_dataset.set_format("numpy")  # type:ignore
        train_dataloader: FlaxDataLoader = train_dataloader.construct(dataset=train_dataset)

        devices = self._get_devices()
        do_distributed: bool = False
        if len(devices) > 1:
            do_distributed = True

        if validation_dataloader is not None:
            validation_dataloader.batch_size *= len(devices)
        train_dataloader.batch_size *= len(devices)

        rng = get_PRNGkey(config.seed)

        if hasattr(model, "params"):
            params = model.params
        else:
            # TODO: Find better way to init the shape
            shape = list(train_dataset["x"].shape)
            shape[0] = 1
            x = jnp.ones(shape)

            params = model.init(rng, x)["params"]

        state = TrainState.create(apply_fn=model.__call__, params=params, tx=optimizer)

        initial_state: Optional[Dict[str, Any]] = None
        if config.state_path.exists():
            logger.info("Recovering from previous run at %s" % config.state_path)
            state = self.load_checkpoint(config.state_path, state)

        if config.train_epochs is None:
            assert config.train_steps is not None
            try:
                train_epochs = len(train_dataloader.dataset) // train_dataloader.batch_size
            except TypeError:
                raise ConfigurationError(
                    "You must set train_epochs for streaming/iterable datasets"
                )

            config.train_epochs = train_epochs

        assert config.train_epochs is not None

        if validation_dataloader is not None:
            if config.validation_steps is None:
                try:
                    config.validation_steps = len(validation_dataloader.dataset)
                except TypeError:
                    raise ConfigurationError(
                        "You must set 'validation_steps' for streaming/iterable datasets"
                    )

        val_metric: Optional[float] = None
        best_val_metric: Optional[float] = None
        start_step: int = 0

        if initial_state is not None:
            val_metric = initial_state[f"val_{config.val_metric_name}"]
            best_val_metric = initial_state[f"best_{config.val_metric_name}"]
            start_step = initial_state["training_epochs"]

        # Initialize callbacks
        callbacks: List[TrainCallback] = [
            callback.construct(
                workspace=workspace,
                train_config=config,
                dataset=dataset,
                train_dataloader=train_dataloader,
                model=model,
                optimizer=optimizer,
                validation_dataloader=validation_dataloader,
            )
            for callback in (callbacks or [])
        ]

        if initial_state:
            for callback, state in zip(callbacks, initial_state["callbacks"]):
                callback.load_state_dict(state)

        del initial_state

        if start_step > 0:
            with Tqdm.tqdm(
                train_dataloader,
                desc=f"Catching dataloader upto step {start_step}",
                total=start_step - 1,
            ) as batch_iter:
                for step, batch in enumerate(batch_iter):
                    del batch
                    if step >= start_step - 1:
                        break

        def train_step(state, batch, dropout_rng):
            # if transformer model
            labels = batch.pop("labels")
            dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)
            grad_fn = jax.value_and_grad(train_wrapper.train_loss)
            loss, grad = grad_fn(state.params, state, batch, dropout_rng, labels)
            if do_distributed:
                grad = jax.lax.pmean(grad, "batch")
            new_state = state.apply_gradients(grads=grad)
            other_metrics = train_wrapper.compute_metrics(state, batch, labels=labels)
            metrics = {"loss": loss}
            metrics.update(other_metrics)
            if do_distributed:
                metrics = jax.lax.pmean(metrics, axis_name="batch")

            return new_state, metrics, new_dropout_rng

        def val_step(state, batch):
            labels = batch.pop("labels")
            logits = state.apply_fn(**batch, params=state.params, train=False)[0]
            metrics = train_wrapper.val_metrics(batch, logits, labels)
            if do_distributed:
                metrics = jax.lax.pmean(metrics, axis_name="batch")
            return metrics

        if do_distributed:
            # NOTE: The trainer currently handles only data parallelism.
            state = jax_utils.replicate(state)
            dropout_rngs = get_multiple_keys(rng, jax.local_device_count())
            parallel_train_step = jax.pmap(train_step, axis_name="batch")
            parallel_val_step = jax.pmap(val_step, axis_name="batch")

        step_per_epoch = train_dataloader.dataset_size // train_dataloader.batch_size
        config.train_steps = step_per_epoch * config.train_epochs

        assert config.train_steps is not None  # for mypy

        for callback in callbacks:
            callback.pre_train_loop()

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {train_dataloader.dataset_size}")
        logger.info(f"  Num Epochs = {config.train_epochs}")
        logger.info(
            f"  Total train batch size (w. parallel & distributed) = {train_dataloader.batch_size}"
        )
        logger.info(f"  Total optimization steps = {config.train_steps}")

        step = start_step

        epochs = Tqdm.tqdm(
            range(config.train_epochs), desc=f"Epoch (1/{config.train_epochs})", position=0
        )

        for epoch in epochs:
            start = time.time()
            train_metrics = []

            for callback in callbacks:
                callback.pre_epoch(step, epoch)

            train_loader = train_dataloader(rng, do_distributed)
            for _ in Tqdm.tqdm(range(step_per_epoch), desc="Training", position=1):
                batch = next(train_loader)
                for callback in callbacks:
                    callback.pre_batch(step, epoch, batch)

                if do_distributed:
                    state, train_metric, dropout_rngs = parallel_train_step(
                        state, batch, dropout_rngs
                    )
                else:
                    state, train_metric, rng = train_step(state, batch, rng)

                train_metrics.append(train_metric)

                for callback in callbacks:
                    callback.post_batch(step, epoch, train_metric)

                if config.should_log_this_step(step):
                    for callback in callbacks:
                        callback.log_batch(step, epoch, train_metric)

                if config.should_checkpoint_this_step(step):
                    self.save_checkpoint(config.state_path, state, step, keep_checkpoints)
                step += 1

                # check if we need to do validation
                if config.validation_split is None:
                    # If we can't validate, we don't.
                    should_validate = False
                elif step == config.train_steps - 1:
                    # If we're at the end of the training run, we always validate.
                    should_validate = True
                elif config.validate_every is not None and step % config.validate_every == 0:
                    # If validate_every is given, we use that to decide.
                    should_validate = True
                else:
                    # Otherwise, we don't validate.
                    should_validate = False

                if should_validate:
                    assert validation_dataloader is not None
                    assert config.validation_steps is not None

                    val_metrics: DefaultDict = defaultdict(list)
                    epoch_eval_metrics: DefaultDict = defaultdict(float)

                    val_dataloader = validation_dataloader(rng, do_distributed)

                    valid_step = 0
                    total_val_steps = len(validation_dataset) // validation_dataloader.batch_size
                    for callback in callbacks:
                        callback.pre_val_loop(step, valid_step, state)

                    for _ in Tqdm.tqdm(range(total_val_steps), desc="Evaluating", position=2):
                        batch = next(val_dataloader)
                        for callback in callbacks:
                            callback.pre_val_batch(step, valid_step, epoch, batch)

                        if do_distributed:
                            metrics = parallel_val_step(state, batch)
                            metrics = jax_utils.unreplicate(metrics)
                        else:
                            metrics = val_step(state, batch)

                        for key, value in metrics.items():
                            val_metrics[key].append(value.item())

                        for callback in callbacks:
                            callback.post_val_batch(step, valid_step, epoch, val_metrics)

                        valid_step += 1

                    for key, value in val_metrics.items():
                        if config.auto_aggregate_val_metric:
                            epoch_eval_metrics[key] = jax.tree_map(
                                jnp.mean, jnp.array(value)
                            ).item()
                        else:
                            epoch_eval_metrics[key] = metrics[key].item()

                    for key, value in epoch_eval_metrics.items():
                        print("Validation %s : %.5f" % (key, value))

                    val_metric = epoch_eval_metrics[config.val_metric_name]

                    assert val_metric is not None

                    if best_val_metric is None:
                        best_val_metric = val_metric
                    elif config.minimize_val_metric and val_metric <= best_val_metric:
                        best_val_metric = val_metric
                    elif not config.minimize_val_metric and val_metric >= best_val_metric:
                        best_val_metric = val_metric

                    for callback in callbacks:
                        callback.post_val_loop(step, epoch, val_metric, best_val_metric)

            if do_distributed:
                train_metric = jax_utils.unreplicate(train_metric)

            for key, value in train_metric.items():
                print("Train %s : %.2f" % (key, value))

            for callback in callbacks:
                callback.post_epoch(step, epoch)

            end = time.time()
            train_time = (end - start) / 60

            desc = f"Epoch... ({epoch + 1}/{config.train_epochs} | Time taken (mins): {train_time})"
            epochs.write(desc)
            epochs.desc = desc

        for callback in callbacks:
            callback.post_train_loop(step, epoch)

        if do_distributed:
            state = jax_utils.unreplicate(state)
        return state

    def save_checkpoint(self, dir: Path, target: PyTree, step: int, keep_checkpoints: int):
        return checkpoints.save_checkpoint(
            dir, target, step, prefix="checkpoint_", keep=keep_checkpoints, overwrite=True
        )

    def load_checkpoint(self, dir: Path, target: PyTree):
        return checkpoints.restore_checkpoint(dir, target, prefix="checkpoint_")

    def _construct_optimizer(self, optimizer):
        self.optimizer = optimizer.construct()
        return self.optimizer

    def _construct_lr_scheduler(self, scheduler):
        self.lr_scheduler = scheduler.construct()
        return self.lr_scheduler

    def _get_devices(self) -> List[Any]:
        device_type = jax.default_backend()
        self.devices = jax.devices()
        device_count = len(self.devices)
        print("Training on %d %s" % (device_count, device_type))
        return self.devices
