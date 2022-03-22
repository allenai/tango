import logging
import os
import shutil
from itertools import islice
from typing import Any, Dict, List, Optional, Set, cast

import more_itertools
import torch
import torch.distributed as dist
from more_itertools import chunked
from torch.utils.data import DistributedSampler

from tango.common.dataset_dict import DatasetDictBase
from tango.common.exceptions import ConfigurationError
from tango.common.lazy import Lazy
from tango.common.tqdm import Tqdm
from tango.common.util import get_extra_imported_modules, import_extra_module
from tango.format import Format
from tango.step import Step

from .data import DataLoader
from .exceptions import StopEarly
from .format import TorchFormat
from .model import Model
from .train_callback import TrainCallback
from .train_config import TrainConfig
from .training_engine import TrainingEngine
from .util import check_dataloader, check_dataset, set_seed_all


@Step.register("torch::train")
class TorchTrainStep(Step):
    """
    A PyTorch training loop step that supports gradient accumulation, distributed training,
    and AMP, with configurable dataloaders, callbacks, optimizer, and LR scheduler.

    .. tip::

        Registered as a :class:`~tango.step.Step` under the name "torch::train".

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
        training_engine: Lazy[TrainingEngine],
        dataset_dict: DatasetDictBase,
        train_dataloader: Lazy[DataLoader],
        *,
        train_split: str = "train",
        validation_split: Optional[str] = None,
        validation_dataloader: Optional[Lazy[DataLoader]] = None,
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
        # Validate device(s).
        if device_count <= 0:
            raise ConfigurationError("Invalid value for 'device_count'. Must be at least 1.")
        devices: List[int]
        if torch.cuda.is_available() and torch.cuda.device_count() >= device_count:
            devices = list(range(device_count))
            self.logger.info("Training on %d GPU%s", device_count, "s" if device_count > 1 else "")
        else:
            devices = [-1] * device_count
            self.logger.info(
                "Training on CPU with %d worker%s", device_count, "s" if device_count > 1 else ""
            )

        if validate_every is not None and validation_split is None:
            raise ConfigurationError(
                "You have set a validation interval, but no validation split. "
                "That's probably unintentional."
            )

        is_distributed = False
        num_workers = 1
        if devices and len(devices) > 1:
            is_distributed = True
            num_workers = len(devices)

        if (train_steps is not None) == (train_epochs is not None):
            raise ConfigurationError(
                "One of 'train_steps' or 'train_epochs' needs to be specified, but not both."
            )

        config = TrainConfig(
            self.unique_id,
            self.work_dir,
            train_split=train_split,
            validation_split=validation_split,
            seed=seed,
            train_steps=train_steps,
            train_epochs=train_epochs,
            grad_accum=grad_accum,
            log_every=log_every,
            checkpoint_every=checkpoint_every,
            validate_every=validate_every,
            validation_steps=validation_steps,
            is_distributed=is_distributed,
            devices=devices,
            distributed_port=distributed_port,
            val_metric_name=val_metric_name,
            minimize_val_metric=minimize_val_metric,
            auto_aggregate_val_metric=auto_aggregate_val_metric,
            remove_stale_checkpoints=remove_stale_checkpoints,
            world_size=num_workers,
        )

        final_model: Model
        if is_distributed:
            import torch.multiprocessing as mp

            mp.spawn(
                _train,
                args=(
                    config,
                    model,
                    training_engine,
                    dataset_dict,
                    train_dataloader,
                    validation_dataloader,
                    callbacks,
                    get_extra_imported_modules(),
                ),
                nprocs=num_workers,
            )
            self.logger.info("Constructing final model")
            final_model = model.construct()
        else:
            final_model = _train(  # type: ignore[assignment]
                0,
                config,
                model,
                training_engine,
                dataset_dict,
                train_dataloader,
                validation_dataloader=validation_dataloader,
                callbacks=callbacks,
            )
            assert final_model is not None
            final_model = final_model.cpu()

        # Load best checkpoint before returning model.
        if config.final_weights_path.is_file():
            self.logger.info(
                f"Loading best weights from {str(config.final_weights_path.resolve())}"
            )
            state = torch.load(config.final_weights_path, map_location="cpu")
            # We use `strict=False` because there might be missing keys due to weight tying.
            final_model.load_state_dict(state, strict=False)

        return final_model


def _train(
    worker_id: int,
    config: TrainConfig,
    model: Lazy[Model],
    training_engine: Lazy[TrainingEngine],
    dataset_dict: DatasetDictBase,
    train_dataloader: Lazy[DataLoader],
    validation_dataloader: Optional[Lazy[DataLoader]] = None,
    callbacks: Optional[List[Lazy[TrainCallback]]] = None,
    include_package: Optional[Set[str]] = None,
) -> Optional[Model]:
    config.worker_id = worker_id

    if config.is_distributed and include_package:
        # During distributed training we need to import `include_package` modules again
        # in order to initialize the lazy objects.
        for package_name in include_package:
            import_extra_module(package_name)

    if config.is_distributed:
        import tango.common.logging as common_logging

        common_logging.initialize_worker_logging(config.worker_id)
    logger = logging.getLogger(TorchTrainStep.__name__)

    training_engine: TrainingEngine = training_engine.construct(
        train_config=config,
        model=model,
    )

    # Check working directory to see if we should recover from a previous run.
    initial_state: Optional[Dict[str, Any]] = None
    if config.state_path.exists():
        if config.is_local_main_process:
            logger.info(f"Recovering from previous run at {str(config.state_path.resolve())}")
        initial_state = training_engine.load_checkpoint(config.state_path)
    device = config.worker_local_default_device

    # Construct data loaders.
    validation_dataloader_: Optional[DataLoader] = None
    if config.validation_split is not None:
        validation_dataset = dataset_dict[config.validation_split]
        check_dataset(validation_dataset, config.validation_split)
        if validation_dataloader is not None:
            validation_dataloader_ = validation_dataloader.construct(dataset=validation_dataset)
        else:
            validation_dataloader_ = train_dataloader.construct(dataset=validation_dataset)
    validation_dataloader: Optional[DataLoader] = validation_dataloader_
    train_dataset = dataset_dict[config.train_split]
    check_dataset(train_dataset, config.train_split)
    train_dataloader: DataLoader = train_dataloader.construct(dataset=train_dataset)

    if config.train_steps is None:
        assert config.train_epochs is not None
        try:
            steps_per_epoch = len(train_dataloader)
        except TypeError:
            raise ConfigurationError("You must set 'train_steps' for streaming/iterable datasets")
        config.train_steps = steps_per_epoch * (config.train_epochs or 1)

    assert config.train_steps is not None  # for mypy

    if validation_dataloader is not None:
        if config.validation_steps is None:
            try:
                config.validation_steps = len(validation_dataloader)
            except TypeError:
                raise ConfigurationError(
                    "You must sest 'validation_steps' for streaming/iterable datasets"
                )

    # Make sure we're using a DistributedSampler during distributed training.
    if config.is_distributed:
        check_dataloader(train_dataloader)
        if validation_dataloader is not None:
            check_dataloader(validation_dataloader)

    # Set random seeds.
    set_seed_all(config.seed)

    batch_loss: float = 0.0
    best_batch_loss: Optional[float] = None
    val_metric: Optional[float] = None
    best_val_metric: Optional[float] = None
    start_step: int = 0
    if initial_state is not None:
        val_metric = initial_state[f"val_{config.val_metric_name}"]
        best_val_metric = initial_state[f"best_{config.val_metric_name}"]
        best_batch_loss = initial_state["best_batch_loss"]
        start_step = initial_state["training_steps"]

    # Initialize callbacks.
    callbacks: List[TrainCallback] = [
        callback.construct(
            train_config=config,
            training_engine=training_engine,
            dataset_dict=dataset_dict,
            train_dataloader=train_dataloader,
            validation_dataloader=validation_dataloader,
        )
        for callback in (callbacks or [])
    ]
    if initial_state:
        for callback, state in zip(callbacks, initial_state["callbacks"]):
            callback.load_state_dict(state)

    del initial_state

    training_engine.model.train()
    training_batches = enumerate(
        islice(
            _cycle_through_epochs(train_dataloader, config.is_distributed, config.grad_accum),
            config.train_steps,
        )
    )

    def is_best_checkpoint() -> bool:
        """
        A closure that we'll call when saving checkpoints to check if we should hardlink
        the best checkpoint path to the current checkpoint file.
        """
        if val_metric is not None and best_val_metric is not None:
            return (config.minimize_val_metric and val_metric <= best_val_metric) or (
                not config.minimize_val_metric and val_metric >= best_val_metric
            )
        elif best_batch_loss is not None:
            return best_batch_loss <= batch_loss
        else:
            return False

    def save_state(step: int):
        """
        A closure that we'll call every `checkpoint_every` steps in the train loop to
        save model and training state.
        """
        train_state = {
            "training_steps": step + 1,
            "best_batch_loss": best_batch_loss,
            f"val_{config.val_metric_name}": val_metric,
            f"best_{config.val_metric_name}": best_val_metric,
            "callbacks": [
                callback.state_dict() for callback in callbacks  # type: ignore[union-attr]
            ],
        }
        # For reason mypy can't figure out that `training_engine` is a `TrainingEngine` in this closure,
        # and not a `Lazy[TrainingEngine]`.
        cast(TrainingEngine, training_engine).save_checkpoint(
            config.state_path_for_step(step), train_state
        )

        # Link to most recent state path.
        # NOTE: While hard linking would be preferable to creating symlinks, some train engines
        # require a whole directory to save their state instead of a single file, which
        # means state_path_for_step will be a directory, so a hard link won't work.
        if config.is_local_main_process:
            if config.state_path.is_symlink():
                config.state_path.unlink()
            config.state_path.symlink_to(
                config.state_path_for_step(step).relative_to(config.work_dir)
            )

            # Link to best state path.
            if is_best_checkpoint():
                if config.best_state_path.is_symlink():
                    config.best_state_path.unlink()
                config.best_state_path.symlink_to(
                    config.state_path_for_step(step).relative_to(config.work_dir)
                )

            # Clean up stale checkpoints.
            if config.remove_stale_checkpoints:
                checkpoints_to_keep = {
                    config.best_state_path.resolve(),
                    config.state_path.resolve(),
                }
                for path in config.work_dir.glob("checkpoint_state_step*"):
                    path = path.resolve()
                    if path not in checkpoints_to_keep:
                        if path.is_file():
                            path.unlink()
                        else:
                            shutil.rmtree(path)

        if config.is_distributed:
            dist.barrier()

    # Catch data loader up to where we left off before.
    current_epoch: int = -1
    if start_step > 0:
        with Tqdm.tqdm(
            training_batches,
            desc=f"Catching dataloader up to step {start_step}",
            total=start_step - 1,
            disable=not config.is_local_main_process,
        ) as batch_iter:
            for step, (current_epoch, batch) in batch_iter:
                del batch
                if step >= start_step - 1:
                    break

    if config.is_distributed:
        dist.barrier()

    for callback in callbacks:
        callback.pre_train_loop()

    train_batch_iterator_tqdm = Tqdm.tqdm(
        training_batches,
        desc="Training",
        initial=start_step,
        total=config.train_steps,
        disable=not config.is_local_main_process,
    )
    train_batch_iterator = more_itertools.peekable(train_batch_iterator_tqdm)
    try:
        for step, (epoch, batch) in train_batch_iterator:
            if epoch != current_epoch:
                # Start of new epoch.
                if epoch > 0:
                    # Call post-epoch callbacks for the last epoch.
                    for callback in callbacks:
                        callback.post_epoch(step, current_epoch)
                for callback in callbacks:
                    callback.pre_epoch(step, epoch)
                current_epoch = epoch

            # Pre-batch callback.
            for callback in callbacks:
                callback.pre_batch(step, current_epoch, batch)
            batch_loss = 0.0
            for micro_batch_idx, micro_batch in enumerate(batch):
                # Get loss.
                micro_batch_loss = training_engine.forward_train(
                    micro_batch, micro_batch_idx, len(batch)
                )
                if torch.isnan(micro_batch_loss):
                    raise ValueError("nan loss encountered")
                batch_loss += micro_batch_loss.detach().item()

                # Calculate gradients.
                training_engine.backward(micro_batch_loss)

                # Clean up in case it saves memory.
                del micro_batch
                del micro_batch_loss

            # Post-batch callback.
            for callback in callbacks:
                callback.post_batch(step, current_epoch, batch_loss)

            if best_batch_loss is None or batch_loss <= best_batch_loss:
                best_batch_loss = batch_loss

            del batch

            training_engine.step()

            # Find out whether we should validate
            if config.validation_split is None:
                # If we can't validate, we don't.
                should_validate = False
            elif step == config.train_steps - 1:
                # If we're at the end of the training run, we always validate.
                should_validate = True
            elif config.validate_every is not None and (step + 1) % config.validate_every == 0:
                # If validate_every is given, we use that to decide.
                should_validate = True
            elif config.validate_every is None and epoch != train_batch_iterator.peek()[1][0]:
                # If validate_every is not given, we validate at the end of the epoch.
                should_validate = True
            else:
                # Otherwise, we don't validate.
                should_validate = False

            # Gather average loss across all workers.
            if (config.should_log_this_step(step) or should_validate) and config.is_distributed:
                batch_loss_tensor = torch.tensor(batch_loss, device=device)
                dist.all_reduce(batch_loss_tensor)
                batch_loss = batch_loss_tensor.item() / config.world_size

            if config.should_log_this_step(step):
                # Callbacks.
                for callback in callbacks:
                    callback.log_batch(step, current_epoch, batch_loss)

                # Update progress bar.
                metrics_to_log: Dict[str, float] = {"batch_loss": batch_loss}
                if val_metric is not None:
                    metrics_to_log[f"val_{config.val_metric_name}"] = val_metric
                if best_val_metric is not None:
                    metrics_to_log[f"best_val_{config.val_metric_name}"] = best_val_metric
                if config.is_local_main_process:
                    train_batch_iterator_tqdm.set_postfix(**metrics_to_log)

            # Validate.
            if should_validate:
                assert validation_dataloader is not None
                assert config.validation_steps is not None

                # Prepare model for validation.
                training_engine.model.eval()

                running_metric = 0.0
                with Tqdm.tqdm(
                    islice(validation_dataloader, config.validation_steps),
                    desc="Validating",
                    total=config.validation_steps,
                    leave=False,
                    disable=not config.is_local_main_process,
                ) as val_batch_iterator:
                    for val_step, val_batch in enumerate(val_batch_iterator):
                        for callback in callbacks:
                            callback.pre_val_batch(step, val_step, current_epoch, val_batch)

                        # Get metric.
                        outputs = training_engine.forward_eval(val_batch)

                        for callback in callbacks:
                            callback.post_val_batch(step, val_step, current_epoch, outputs)
                        metric = outputs[config.val_metric_name]

                        if config.auto_aggregate_val_metric:
                            running_metric += metric if isinstance(metric, float) else metric.item()
                            val_metric = running_metric / (val_step + 1)
                        else:
                            val_metric = metric if isinstance(metric, float) else metric.item()

                        # Average metric across all workers.
                        if (
                            config.is_distributed
                            and config.should_log_this_val_step(val_step)
                            and config.auto_aggregate_val_metric
                        ):
                            val_metric_tensor = torch.tensor(val_metric, device=device)
                            dist.all_reduce(val_metric_tensor)
                            val_metric = val_metric_tensor.item() / config.world_size

                        # Update progress bar.
                        if config.is_local_main_process and config.should_log_this_val_step(
                            val_step
                        ):
                            val_batch_iterator.set_postfix(**{config.val_metric_name: val_metric})

                        # Clean up.
                        del val_batch
                        del outputs
                        del metric

                assert val_metric is not None

                # Reset model to train mode.
                training_engine.model.train()

                if best_val_metric is None:
                    best_val_metric = val_metric
                elif config.minimize_val_metric and val_metric <= best_val_metric:
                    best_val_metric = val_metric
                elif not config.minimize_val_metric and val_metric >= best_val_metric:
                    best_val_metric = val_metric

                # Post validation callback.
                for callback in callbacks:
                    callback.post_val_loop(step, current_epoch, val_metric, best_val_metric)

                # Update progress bar again.
                metrics_to_log = {
                    "batch_loss": batch_loss,
                    f"val_{config.val_metric_name}": val_metric,
                    f"best_{config.val_metric_name}": best_val_metric,
                }
                if config.is_local_main_process:
                    train_batch_iterator_tqdm.set_postfix(**metrics_to_log)

            # Checkpoint.
            if config.should_checkpoint_this_step(step):
                save_state(step)

        # End train loop

        # Final post-epoch callback.
        for callback in callbacks:
            callback.post_epoch(step, current_epoch)
    except StopEarly:
        if config.is_local_main_process:
            logger.info("Stopping early!")
    finally:
        train_batch_iterator_tqdm.close()

    if config.is_distributed:
        dist.barrier()

    for callback in callbacks:
        callback.post_train_loop(step, current_epoch)

    if config.is_local_main_process:
        # It's possible this file already exists if the step previously failed after
        # already saving the final weights.
        if config.final_weights_path.is_file():
            os.remove(config.final_weights_path)
        training_engine.save_complete_weights_from_checkpoint(
            config.best_state_path, config.final_weights_path
        )

    if not config.is_distributed:
        return training_engine.model
    else:
        return None


def _cycle_through_epochs(dataloader: DataLoader, is_distributed: bool, grad_accum: int):
    epoch = 0
    while True:
        if is_distributed and isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(epoch)
        for batch in chunked(dataloader, grad_accum):
            yield epoch, batch
        epoch += 1
