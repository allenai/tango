# import logging
from typing import Any, Dict, List, Optional, Union

import flax.jax_utils
import jax
import jax.numpy as jnp
from flax.training.common_utils import get_metrics
from tqdm import tqdm

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
from .train_state import FlaxTrainState
from .util import get_PRNGkey


@Step.register("flax::train")
class FlaxTrainStep(Step):
    """
    A Flax training step.
    """

    DETERMINISTIC = True
    CACHEABLE = True
    FORMAT: Format = FlaxFormat()

    def run(
        self,
        model: Model,
        dataset: Union[DatasetDictBase],
        optimizer: Lazy[Optimizer],
        train_dataloader: Lazy[FlaxDataLoader],
        *,
        lr_scheduler: Optional[Lazy[LRScheduler]] = None,
        train_split: Optional[str] = "train",
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
    ) -> Model:
        """
        Run a basic training loop to train the ``model``.

        # :param model:
        #     The model to train.
        """

        return self._train(
            dataset=dataset,
            model=model,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            lr_scheduler=lr_scheduler,
            train_split=train_split,
            validation_split=validation_split,
            validation_dataloader=validation_dataloader,
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
        dataset: Union[DatasetDictBase],
        train_dataloader: Lazy[FlaxDataLoader],
        *,
        lr_scheduler: Optional[Lazy[LRScheduler]],
        train_split: Optional[str] = "train",
        validation_split: Optional[str] = "val",
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
    ) -> Model:

        if validate_every is not None and validation_split is None:
            raise ConfigurationError(
                "You have set a validation interval, but no validation split. "
                "That's probably unintentional."
            )

        if (train_steps is not None) == (train_epochs is not None):
            raise ConfigurationError(
                "One of 'train_steps' or 'train_epochs' needs to be specified, but not both."
            )

        if isinstance(dataset, DatasetDictBase) and train_split is None:
            raise ConfigurationError("Specify the train split for Datasets object.")

        config = TrainConfig(
            self.unique_id,
            self.work_dir,
            step_name=self.name,
            train_split="train",
            validation_split="val",
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
            dataset,
            train_dataloader,
            validation_dataloader,
            callbacks,
        )
        assert final_model is not None

        # return best checkpoint
        return final_model

    def train_helper(
        self,
        workspace: Workspace,
        config: TrainConfig,
        model: Model,
        optimizer: Optimizer,
        lr_scheduler: Optional[LRScheduler],
        dataset: DatasetDictBase,
        train_dataloader: Lazy[FlaxDataLoader],
        validation_dataloader: Optional[Lazy[FlaxDataLoader]] = None,
        callbacks: Optional[List[Lazy[TrainCallback]]] = None,
    ) -> Model:

        # logger = logging.getLogger(FlaxTrainStep.__name__)

        train_state = FlaxTrainState(model, optimizer)

        initial_state: Optional[Dict[str, Any]] = None

        # TODO: recover from a previous run

        # construct data loaders
        validation_dataloader_: Optional[FlaxDataLoader] = None
        if config.validation_split is not None:
            validation_dataset = dataset[config.validation_split]
            if validation_dataloader is not None:
                validation_dataloader_ = validation_dataloader.construct(dataset=validation_dataset)
            else:
                validation_dataloader_ = train_dataloader.construct(dataset=validation_dataset)

        validation_dataloader = validation_dataloader_
        train_dataset = dataset[config.train_split]
        train_dataloader: FlaxDataLoader = train_dataloader.construct(dataset=train_dataset)

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
        val_metric: float = 0.0
        best_val_metric: float = 0.0
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
                validation_dataloader=validation_dataloader,
            )
            for callback in (callbacks or [])
        ]

        if initial_state:
            for callback, state in zip(callbacks, initial_state["callbacks"]):
                callback.load_state_dict(state)

        del initial_state

        # Catch data loader up to where we left off before
        # current_step: int = -1
        if start_step > 0:
            with Tqdm.tqdm(
                train_dataloader,
                desc=f"Catching dataloader upto step {start_step}",
                total=start_step - 1,
            ) as batch_iter:
                for step, (epoch, batch) in batch_iter:
                    del batch
                    if step >= start_step - 1:
                        break

        # replicate state across devices
        train_state.replicate_state()
        parallel_train_step = jax.pmap(self.train_step, axis_name="batch", donate_argnums=(0,))
        parallel_val_step = jax.pmap(self.val_step, axis_name="batch")

        step_per_epoch = len(train_dataloader.dataset) // train_dataloader.batch_size
        config.train_steps = step_per_epoch * config.train_epochs

        for callback in callbacks:
            callback.pre_train_loop()

        rng, input_rng = get_PRNGkey()
        dropout_rngs = jax.random.split(rng, jax.local_device_count())

        epochs = tqdm(
            range(config.train_epochs), desc=f"Epoch (1/{config.train_epochs})", position=0
        )

        for epoch in epochs:
            train_metrics = []

            for callback in callbacks:
                callback.pre_epoch(step, epoch)

            batches = tqdm(train_dataloader, total=step_per_epoch, desc="Training", position=1)

            for step, batch in enumerate(batches):
                for callback in callbacks:
                    callback.pre_batch(step, epoch, batch)

                train_metric, dropout_rngs = parallel_train_step(train_state, batch, dropout_rngs)
                train_metrics.append(train_metric)

                for callback in callbacks:
                    callback.post_batch(step, epoch, batch_loss)

                if config.should_log_this_step(step):
                    for callback in callbacks:
                        callback.log_batch(step, epoch, batch_loss)

            for callback in callbacks:
                callback.post_epoch(step, epoch)

            # check if we need to do validation
            if config.validation_split is None:
                # If we can't validate, we don't.
                should_validate = False
            elif step == config.train_steps - 1:
                # If we're at the end of the training run, we always validate.
                should_validate = True
            elif config.validate_every is not None and (step + 1) % config.validate_every == 0:
                # If validate_every is given, we use that to decide.
                should_validate = True
            elif config.validate_every is None and epoch != batches.peek()[1][0]:
                # If validate_every is not given, we validate at the end of the epoch.
                should_validate = True
            else:
                # Otherwise, we don't validate.
                should_validate = False

            if should_validate:
                assert validation_dataloader is not None
                assert config.validation_steps is not None

                # validation
                val_metrics = []

                validation_dataloader_tqdm = tqdm(
                    validation_dataloader(input_rng),
                    total=len(validation_dataset) // validation_dataloader.batch_size,
                    desc="Evaluating",
                    position=2,
                )

                for val_step, batch in enumerate(validation_dataloader_tqdm):
                    for callback in callbacks:
                        callback.pre_val_batch(step, val_step, epoch, batch)

                    val_metrics.append(parallel_val_step(state, batch))

                    # store and checkpoint

                    for callback in callbacks:
                        callback.post_val_batch(step, val_step, epoch, batch)

                epoch_eval_metrics = flax.jax_utils.unreplicate(val_metrics)
                epoch_eval_metrics = get_metrics(val_metrics)
                epoch_eval_metrics = jax.tree_map(jnp.mean, epoch_eval_metrics)

                for callback in callbacks:
                    callback.post_val_loop(step, epoch, val_metric, best_val_metric)

            train_metrics = flax.jax_utils.unreplicate(train_metrics)
            epoch_train_metrics = get_metrics(train_metrics)
            epoch_train_metrics = jax.tree_map(jnp.mean, epoch_train_metrics)

        for callback in callbacks:
            callback.post_train_loop(step, epoch)

        # load best checkpoint and return
        return model

    def train_step(self, train_state, batch, dropout_rng):
        dropout_rng, new_dropout_rng = get_PRNGkey(dropout_rng)
        loss_fn = None  # TODO
        loss = train_state.update_step(batch, loss_fn)
        metrics = jax.lax.pmean({"loss": loss}, axis_name="batch")
        return metrics, new_dropout_rng

    def val_step(self, train_state, batch):
        logits_fn = None  # TODO
        loss = train_state.eval_state(batch, logits_fn)
        metrics = jax.lax.pmean({"loss": loss}, axis_name="batch")
        return metrics

    def save_checkpoint(self):
        raise NotImplementedError

    def load_checkpoint(self):
        raise NotImplementedError

    def is_best_checkpoint(self) -> bool:
        raise NotImplementedError

    def _construct_optimizer(self, optimizer):
        self.optimizer = optimizer.construct()
        return self.optimizer

    def _construct_lr_scheduler(self, scheduler):
        self.lr_scheduler = scheduler.construct()
        return self.lr_scheduler
