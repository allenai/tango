import logging
from typing import Any, Dict, List, Optional, Union, DefaultDict
from collections import defaultdict

import jax
import jax.numpy as jnp
from flax.training import checkpoints

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
from .train_state import FlaxTrainState, FlaxTrainWrapper
from .util import get_multiple_keys, get_PRNGkey

PyTree = Any


@Step.register("flax::train")
class FlaxTrainStep(Step):
    """
    A Flax training step.
    """

    DETERMINISTIC = True
    CACHEABLE = True
    FORMAT: Format = FlaxFormat()
    SKIP_ID_ARGUMENTS = {"log_every"}

    def run(  # type: ignore[override]
        self,
        model: Model,
        dataset: Union[DatasetDictBase, Dict],
        optimizer: Lazy[Optimizer],
        train_dataloader: Lazy[FlaxDataLoader],
        *,
        train_wrapper: FlaxTrainWrapper,
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
        :param train_wrapper
            A Wrapper class that defines ``loss_fn()``, ``eval_fn()`` and ``compute_metrics()``
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
            The trained model with the best checkpoint loaded.
        """

        return self._train(
            dataset=dataset,
            model=model,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            train_wrapper=train_wrapper,
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
        dataset: Union[DatasetDictBase, Dict],
        train_dataloader: Lazy[FlaxDataLoader],
        *,
        train_wrapper: FlaxTrainWrapper,
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
            seed=42,
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
        lr_scheduler: Optional[LRScheduler],
        train_wrapper: FlaxTrainWrapper,
        dataset: Union[DatasetDictBase, Dict],
        train_dataloader: Lazy[FlaxDataLoader],
        validation_dataloader: Optional[Lazy[FlaxDataLoader]] = None,
        callbacks: Optional[List[Lazy[TrainCallback]]] = None,
    ) -> PyTree:

        logger = logging.getLogger(FlaxTrainStep.__name__)

        # construct data loaders
        validation_dataloader_: Optional[FlaxDataLoader] = None
        if config.validation_split is not None:
            validation_dataset = dataset[config.validation_split]
            if validation_dataloader is not None:
                validation_dataloader_ = validation_dataloader.construct(dataset=validation_dataset)
            else:
                validation_dataloader_ = train_dataloader.construct(dataset=validation_dataset)

        validation_dataloader = validation_dataloader_

        train_dataset = dataset
        if config.train_split is not None:
            train_dataset = dataset[config.train_split]
        train_dataloader: FlaxDataLoader = train_dataloader.construct(dataset=train_dataset)

        devices = self._get_devices()
        do_distributed: bool = False
        if len(devices) > 1:
            do_distributed = True

        flax_state = FlaxTrainState(
            model,
            optimizer,
            train_wrapper,
            do_distributed=do_distributed,
            lr_scheduler=lr_scheduler,
            shape=[1, 28, 28, 1],
        )

        initial_state: Optional[Dict[str, Any]] = None
        if config.state_path.exists():
            logger.info("Recovering from previous run at %s" % config.state_path)
            flax_state.state = self.load_checkpoint(config.state_path, flax_state.state)

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

        batch_loss: float = 0.0
        # best_batch_loss: Optional[float] = None
        val_metric: Optional[float] = None
        best_val_metric: Optional[float] = None
        start_step: int = 0

        if initial_state is not None:
            val_metric = initial_state[f"val_{config.val_metric_name}"]
            best_val_metric = initial_state[f"best_{config.val_metric_name}"]
            # best_batch_loss = initial_state["best_batch_loss"]
            start_step = initial_state["training_epochs"]

        # Initialize callbacks
        callbacks: List[TrainCallback] = [
            callback.construct(
                workspace=workspace,
                train_config=config,
                dataset=dataset,
                train_dataloader=train_dataloader,
                validation_dataloader=validation_dataloader,
            )
            for callback in (callbacks or [])
        ]

        if initial_state:
            for callback, state in zip(callbacks, initial_state["callbacks"]):
                callback.load_state_dict(state)

        del initial_state

        # Catch data loader up to where we left off before
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

        print("device count: ", jax.local_device_count())
        rng = get_PRNGkey()
        dropout_rngs = rng[1]
        if do_distributed:
            dropout_rngs = get_multiple_keys(rng, jax.local_device_count())
            flax_state.replicate_state()  # check this
            parallel_train_step = jax.pmap(
                flax_state.train_state, axis_name="batch", donate_argnums=(0,)
            )
            parallel_val_step = jax.pmap(flax_state.val_state, axis_name="batch")

        step_per_epoch = train_dataloader.dataset_size // train_dataloader.batch_size
        step_per_epoch = 100
        config.train_steps = step_per_epoch * config.train_epochs

        for callback in callbacks:
            callback.pre_train_loop()

        epochs = Tqdm.tqdm(
            range(config.train_epochs), desc=f"Epoch (1/{config.train_epochs})", position=0
        )
        for epoch in epochs:
            # train_metrics: Dict = {"loss": [], "acc": []}
            train_metrics: DefaultDict = defaultdict(list)

            for callback in callbacks:
                callback.pre_epoch(step, epoch)

            batches = Tqdm.tqdm(
                train_dataloader(rng, do_distributed),
                initial=start_step,
                total=step_per_epoch,
                position=1,
                desc="Training",
            )

            step = start_step
            for batch in batches:
                for callback in callbacks:
                    callback.pre_batch(step, epoch, batch)

                if do_distributed:
                    train_metric = parallel_train_step(batch, dropout_rngs)
                else:
                    train_metric = flax_state.train_state(batch, rng)

                for key, value in train_metric.items():
                    train_metrics[key].append(value)

                for callback in callbacks:
                    callback.post_batch(step, epoch, batch_loss)

                if config.should_log_this_step(step):
                    for callback in callbacks:
                        callback.log_batch(step, epoch, batch_loss)
                del batch

                if config.should_checkpoint_this_step(step):
                    self.save_checkpoint(config.work_dir, flax_state.state, step)
                step += 1

            # TODO: Need to unreplicate state
            # if do_distributed:
            #     train_metrics = flax.jax_utils.unreplicate(train_metrics)

            epoch_train_metrics: DefaultDict = defaultdict(float)

            for key, value in train_metrics.items():
                epoch_train_metrics[key] = jax.tree_map(jnp.mean, jnp.array(value)).item()

            print("Train metrics " , epoch_train_metrics)

            for callback in callbacks:
                callback.post_epoch(step, epoch)

            # check if we need to do validation
            if config.validation_split:
                should_validate = True
            elif config.validation_split is None:
                # If we can't validate, we don't.
                should_validate = False
            elif step == config.train_steps - 1:
                # If we're at the end of the training run, we always validate.
                should_validate = True
            elif config.validate_every is not None and (step + 1) % config.validate_every == 0:
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

                validation_dataloader_tqdm = Tqdm.tqdm(
                    validation_dataloader(rng, do_distributed),
                    total=len(validation_dataset) // validation_dataloader.batch_size,
                    desc="Evaluating",
                    position=2,
                )

                for val_step, batch in enumerate(validation_dataloader_tqdm):
                    for callback in callbacks:
                        callback.pre_val_batch(step, val_step, epoch, batch)

                    if do_distributed:
                        metrics = parallel_val_step(batch)
                    else:
                        metrics = flax_state.val_state(batch)

                    for key, value in metrics.items():
                        val_metrics[key].append(value)

                    for callback in callbacks:
                        callback.post_val_batch(step, val_step, epoch, batch)

                for key, value in val_metrics.items():
                    epoch_eval_metrics = jax.tree_map(jnp.mean, jnp.array(value)).item()

                # TODO: Need to unreplicate state for distributed training

                for callback in callbacks:
                    callback.post_val_loop(step, epoch, val_metric, best_val_metric)

                print("Validation metrics " , epoch_eval_metrics)

            epochs.desc = f"Epoch ... {epoch + 1}/{config.train_epochs}"

        for callback in callbacks:
            callback.post_train_loop(step, epoch)

        # TODO: Load the best checkpoint
        return flax_state.state

    def save_checkpoint(self, dir, target, step) -> str:
        return checkpoints.save_checkpoint(
            dir, target, step, prefix="checkpoint_", keep=100, overwrite=True
        )

    def load_checkpoint(self, dir, target):
        dir = self.config.work_dir
        return checkpoints.restore_checkpoint(dir, target, prefix="checkpoint_")

    def is_best_checkpoint(self) -> bool:
        raise NotImplementedError

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
