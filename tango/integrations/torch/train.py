import random
import warnings
from itertools import islice
from typing import Any, Dict, List, Optional, cast

import numpy as np
import torch
import torch.distributed as dist
from more_itertools import chunked
from torch.utils.data import DistributedSampler, IterableDataset

from tango.common.dataset_dict import DatasetDictBase
from tango.common.exceptions import ConfigurationError
from tango.common.lazy import Lazy
from tango.common.tqdm import Tqdm
from tango.format import Format
from tango.step import Step

from .accelerator import Accelerator, DefaultAccelerator
from .data import DataLoader
from .exceptions import StopEarly
from .format import TorchFormat
from .model import Model
from .optim import LRScheduler, Optimizer
from .train_callback import TrainCallback
from .train_config import TrainConfig


@Step.register("torch::train")
class TorchTrainStep(Step):
    """
    A basic PyTorch training loop step that supports gradient accumulation, distributed training,
    and AMP, with configurable dataloaders, callbacks, optimizer, and LR scheduler.

    .. tip::

        Registered as a :class:`~tango.step.Step` under the name "torch::train".

    .. warning::

        During validation, the validation metric (specified by the ``val_metric_name`` parameter)
        is aggregated by simply averaging across validation batches and distributed processes.
        This behavior is usually correct when your validation metric is "loss" or "accuracy",
        for example, but may not be correct for other metrics such as F1.

        If this is not correct for your metric you will need to handle the aggregation
        internally in your model so that the metric returned from :meth:`torch.nn.Module.forward()`
        is already aggregated correctly. Then set the parameter ``aggregate_val_metric`` to
        ``False``.

        Note that correctly aggregating your metric during distributed training will
        involve distributed communication.

    """

    DETERMINISTIC: bool = True
    CACHEABLE = True
    FORMAT: Format = TorchFormat()

    def run(  # type: ignore[override]
        self,
        model: Lazy[Model],
        dataset_dict: DatasetDictBase,
        train_dataloader: Lazy[DataLoader],
        optimizer: Lazy[Optimizer],
        *,
        train_split: str = "train",
        validation_split: Optional[str] = None,
        lr_scheduler: Optional[Lazy[LRScheduler]] = None,
        validation_dataloader: Optional[Lazy[DataLoader]] = None,
        seed: int = 42,
        train_steps: Optional[int] = None,
        validation_steps: Optional[int] = None,
        grad_accum: int = 1,
        log_every: int = 10,
        checkpoint_every: int = 100,
        validate_every: int = 100,
        amp: bool = False,
        max_grad_norm: Optional[float] = None,
        devices: Optional[List[int]] = None,
        distributed_port: str = "54761",
        val_metric_name: str = "loss",
        minimize_val_metric: bool = True,
        aggregate_val_metric: bool = True,
        callbacks: Optional[List[Lazy[TrainCallback]]] = None,
        accelerator: Lazy[Accelerator] = Lazy(DefaultAccelerator),
        remove_stale_checkpoints: bool = True,
    ) -> Model:
        """
        Run a basic training loop to train the ``model``.

        Parameters
        ----------

        model : :class:`Model`
            The model to train. It should return a ``dict`` that includes the ``loss``
            during training and validation.
        dataset_dict : :class:`~tango.common.dataset_dict.DatasetDictBase`
            The train and optional validation data.
        train_dataloader : :class:`DataLoader`
            The data loader that generates training batches. The batches should be :class:`dict`
            objects.
        optimizer : :class:`Optimizer`
            A PyTorch :class:`~torch.optim.Optimizer`.
        train_split : :class:`str`, optional
            The name of the data split used for training in the ``dataset_dict``.
            Default is "train".
        validation_split : :class:`str`, optional
            Optional name of the validation split in the ``dataset_dict``. Default is ``None``,
            which means no validation.
        lr_scheduler : :class:`LRScheduler`, optional
            An optional
            `learning rate scheduler <https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate>`_
            to adjust the learning rate throughout training.
        validation_dataloader : :class:`DataLoader`, optional
            An optional data loader for generating validation batches. The batches should be
            :class:`dict` objects. If not specified, but ``validation_split`` is given,
            the validation ``DataLoader`` will be constructed from the same parameters
            as the train ``DataLoader``.
        seed : :class:`int`, optional
            Used to set the RNG states at the beginning of training.
        train_steps : :class:`int`, optional
            The number of steps to train for. If not specified training will
            stop after a complete iteration through the `train_dataloader`.
        validation_steps : :class:`int`, optional
            The number of steps to validate for. If not specified validation
            will stop after a complete iteration through the `validation_dataloader`.
        grad_accum : :class:`int`, optional
            The number of gradient accumulation steps. Defaults to 1.

            .. note::
                This parameter - in conjuction with the settings of your data loader
                and the number distributed workers -
                determines the *effective batch size* of your training run.

        log_every : :class:`int`, optional
            Log every this many steps.
        checkpoint_every : :class:`int`, optional
            Save a checkpoint every this many steps.
        validate_every : :class:`int`, optional
            Run the validation loop every this many steps.
        devices : ``List[int]``, optional
            The IDs of the CUDA devices to train on.
        distributed_port : :class:`str`
            The port of the distributed process group. Default = "54761".
        val_metric_name : :class:`str`
            The name of the validation metric, i.e. the key of the metric in the dictionary
            returned by the forward pass of the model. Default is "loss".
        minimize_val_metric : :class:`bool`
            Whether the validation metric is meant to be minimized (such as the loss).
            Default is ``True``. When using a metric such as accuracy, you should set
            this to ``False``.
        aggregate_val_metric : :class:`bool`
            If ``True`` (the default), the validation metric will be averaged across
            validation batches and distributed processes. This may not be the correct
            behavior for some metrics (such as F1), in which you should set this to
            ``False`` and handle the aggregation internally in your model.
        callbacks : ``List[TrainCallback]``
            A list of :class:`TrainCallback`.
        accelerator : :class:`Accelerator`
            An :class:`Accelerator` to use. By default :class:`DefaultAccelerator` is used.
        remove_stale_checkpoints : :class:`bool`
            If ``True`` (the default), stale checkpoints will be removed throughout training so that
            only the latest and best checkpoints are kept.

        Returns
        -------
        :class:`Model`
            The trained model on CPU with the weights from the best checkpoint loaded.

        """
        # Validate device(s).
        if torch.cuda.is_available():
            if devices is None:
                print("CUDA is available")
            elif all((x >= 0 for x in devices)):
                if torch.cuda.device_count() < len(set(devices)):
                    raise ConfigurationError(
                        f"Only found {torch.cuda.device_count()} CUDA devices, "
                        f"but you specified {len(set(devices))} device IDs"
                    )
            elif not all((x == -1 for x in devices)):
                raise ConfigurationError("Invalid value for 'devices'")
        else:
            if devices and not all((x == -1 for x in devices)):
                raise ConfigurationError(
                    "CUDA not found, so only '-1' allowed for device IDs, found {devices}"
                )
            if amp:
                raise ConfigurationError("AMP requires CUDA")

        is_distributed = False
        num_workers = 1
        if devices and len(devices) > 1:
            is_distributed = True
            num_workers = len(devices)

        config = TrainConfig(
            self.work_dir,
            train_split=train_split,
            validation_split=validation_split,
            seed=seed,
            train_steps=train_steps,
            grad_accum=grad_accum,
            log_every=log_every,
            checkpoint_every=checkpoint_every,
            validate_every=validate_every,
            is_distributed=is_distributed,
            val_metric_name=val_metric_name,
            minimize_val_metric=minimize_val_metric,
            aggregate_val_metric=aggregate_val_metric,
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
                    dataset_dict,
                    train_dataloader,
                    optimizer,
                    lr_scheduler,
                    validation_dataloader,
                    callbacks,
                    accelerator,
                    self.executor.include_package,
                ),
                nprocs=num_workers,
            )
            print("Constructing final model")
            final_model = model.construct()
        else:
            final_model = _train(  # type: ignore[assignment]
                0,
                config,
                model,
                dataset_dict,
                train_dataloader,
                optimizer,
                lr_scheduler=lr_scheduler,
                validation_dataloader=validation_dataloader,
                callbacks=callbacks,
                accelerator=accelerator,
            )
            assert final_model is not None
            final_model = final_model.cpu()

        # Load best checkpoint before returning model.
        if config.final_weights_path.is_file():
            print(f"Loading best weights from {config.final_weights_path.resolve().name}")
            state = torch.load(config.final_weights_path, map_location="cpu")
            final_model.load_state_dict(state)

        return final_model


def _train(
    worker_id: int,
    config: TrainConfig,
    model: Lazy[Model],
    dataset_dict: DatasetDictBase,
    train_dataloader: Lazy[DataLoader],
    optimizer: Lazy[Optimizer],
    lr_scheduler: Optional[Lazy[LRScheduler]] = None,
    validation_dataloader: Optional[Lazy[DataLoader]] = None,
    callbacks: Optional[List[Lazy[TrainCallback]]] = None,
    accelerator: Lazy[Accelerator] = Lazy(DefaultAccelerator),
    include_package: Optional[List[str]] = None,
) -> Optional[Model]:
    config.worker_id = worker_id

    if config.is_distributed and include_package:
        # During distributed training we need to import `include_package` modules again
        # in order to initialize the lazy objects.
        from tango.common.util import import_module_and_submodules

        for package_name in include_package:
            import_module_and_submodules(package_name)

    if config.is_distributed:
        import tango.common.logging as common_logging

        common_logging.initialize_logging(prefix=f"[worker {worker_id}]")

    # Check working directory to see if we should recover from a previous run.
    initial_state: Optional[Dict[str, Any]] = None
    if config.state_path.is_file():
        if config.is_local_main_process:
            print(f"Recovering from previous run at {str(config.state_path.name)}")
        initial_state = torch.load(config.state_path)

    accelerator: Accelerator = accelerator.construct(
        train_config=config,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        initial_state=initial_state,
    )
    device = config.worker_local_default_device

    # Construct data loaders.
    validation_dataloader_: Optional[DataLoader] = None
    if config.validation_split is not None:
        validation_dataset = dataset_dict[config.validation_split]
        _check_dataset(validation_dataset, config.validation_split)
        if validation_dataloader is not None:
            validation_dataloader_ = validation_dataloader.construct(dataset=validation_dataset)
        else:
            validation_dataloader_ = train_dataloader.construct(dataset=validation_dataset)
    validation_dataloader: Optional[DataLoader] = validation_dataloader_
    train_dataset = dataset_dict[config.train_split]
    _check_dataset(train_dataset, config.train_split)
    train_dataloader: DataLoader = train_dataloader.construct(dataset=train_dataset)

    if config.train_steps is None:
        try:
            config.train_steps = len(train_dataloader)
        except TypeError:
            raise ConfigurationError("You must sest 'train_steps' for streaming/iterable datasets")
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
        _check_dataloader(train_dataloader)
        if validation_dataloader is not None:
            _check_dataloader(validation_dataloader)

    # Set random seeds.
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

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
            accelerator=accelerator,
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

    accelerator.model.train()
    training_batches = enumerate(
        islice(
            chunked(
                _cycle_through_epochs(train_dataloader, config.is_distributed), config.grad_accum
            ),
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
        # For reason mypy can't figure out that `accelerator` is an `Accelerator` in this closure,
        # and not a `Lazy[Accelerator]`.
        cast(Accelerator, accelerator).save_checkpoint(
            config.state_path_for_step(step), train_state
        )

        # Link to most recent state path.
        # NOTE: While hard linking would be preferable to creating symlinks, some accelerators
        # require a whole directory to save their state instead of a single file, which
        # means state_path_for_step will be a directory, so a hard link won't work.
        if config.state_path.is_symlink():
            config.state_path.unlink()
        config.state_path.symlink_to(config.state_path_for_step(step))

        # Link to best state path.
        if is_best_checkpoint():
            if config.best_state_path.is_symlink():
                config.best_state_path.unlink()
            config.best_state_path.symlink_to(config.state_path_for_step(step))

        # Clean up stale checkpoints.
        if config.remove_stale_checkpoints:
            checkpoints_to_keep = {
                config.best_state_path.resolve(),
                config.state_path.resolve(),
            }
            for path in config.work_dir.glob(f"state_worker{worker_id}_*.pt"):
                path = path.resolve()
                if path not in checkpoints_to_keep:
                    path.unlink()

    # Catch data loader up to where we left off before.
    if start_step > 0:
        with Tqdm.tqdm(
            training_batches,
            desc=f"Catching dataloader up to step {start_step}",
            total=start_step - 1,
            disable=not config.is_local_main_process,
        ) as batch_iter:
            for step, batch in batch_iter:
                del batch
                if step >= start_step - 1:
                    break

    if config.is_distributed:
        dist.barrier()

    for callback in callbacks:
        callback.pre_train_loop()

    train_batch_iterator = Tqdm.tqdm(
        training_batches,
        desc="Training",
        initial=start_step,
        total=config.train_steps,
        disable=not config.is_local_main_process,
    )
    try:
        for step, batch in train_batch_iterator:
            for callback in callbacks:
                callback.pre_batch(step, batch)
            batch_loss = 0.0
            for micro_batch_idx, micro_batch in enumerate(batch):
                # Get loss.
                micro_batch_loss = accelerator.forward_train(
                    micro_batch, micro_batch_idx, len(batch)
                )
                if torch.isnan(micro_batch_loss):
                    raise ValueError("nan loss encountered")
                batch_loss += micro_batch_loss.detach().item()

                # Calculate gradients.
                accelerator.backward(micro_batch_loss)

                # Clean up in case it saves memory.
                del micro_batch
                del micro_batch_loss

            for callback in callbacks:
                callback.post_batch(step, batch_loss)

            if best_batch_loss is None or batch_loss <= best_batch_loss:
                best_batch_loss = batch_loss

            del batch

            accelerator.step()

            # Gather average loss across all workers.
            if (
                config.should_log_this_step(step) or config.should_validate_this_step(step)
            ) and config.is_distributed:
                batch_loss_tensor = torch.tensor(batch_loss, device=device)
                dist.all_reduce(batch_loss_tensor)
                batch_loss = batch_loss_tensor.item() / config.world_size

            if config.should_log_this_step(step):
                # Callbacks.
                for callback in callbacks:
                    callback.log_batch(step, batch_loss)

                # Update progress bar.
                metrics_to_log: Dict[str, float] = {"batch_loss": batch_loss}
                if val_metric is not None:
                    metrics_to_log[f"val_{config.val_metric_name}"] = val_metric
                if best_val_metric is not None:
                    metrics_to_log[f"best_val_{config.val_metric_name}"] = best_val_metric
                if config.is_local_main_process:
                    train_batch_iterator.set_postfix(**metrics_to_log)

            # Validate.
            if config.should_validate_this_step(step):
                assert validation_dataloader is not None
                assert config.validation_steps is not None

                # Prepare model for validation.
                accelerator.model.eval()

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
                            callback.pre_val_batch(step, val_step, val_batch)
                        # Get metric.
                        outputs = accelerator.forward_eval(val_batch)

                        for callback in callbacks:
                            callback.post_val_batch(step, val_step, outputs)
                        metric = outputs[config.val_metric_name]

                        if config.aggregate_val_metric:
                            running_metric += metric if isinstance(metric, float) else metric.item()
                            val_metric = running_metric / (val_step + 1)
                        else:
                            val_metric = metric if isinstance(metric, float) else metric.item()

                        # Average metric across all workers.
                        if (
                            config.is_distributed
                            and config.should_log_this_val_step(val_step)
                            and config.aggregate_val_metric
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
                accelerator.model.train()

                if best_val_metric is None:
                    best_val_metric = val_metric
                elif config.minimize_val_metric and val_metric <= best_val_metric:
                    best_val_metric = val_metric
                elif not config.minimize_val_metric and val_metric >= best_val_metric:
                    best_val_metric = val_metric

                for callback in callbacks:
                    callback.post_val_loop(step, val_metric, best_val_metric)

                # Update progress bar again.
                metrics_to_log = {
                    "batch_loss": batch_loss,
                    f"val_{config.val_metric_name}": val_metric,
                    f"best_{config.val_metric_name}": best_val_metric,
                }
                if config.is_local_main_process:
                    train_batch_iterator.set_postfix(**metrics_to_log)

            # Checkpoint.
            if config.should_checkpoint_this_step(step):
                save_state(step)
    except StopEarly:
        if config.is_local_main_process:
            print("Stopping early!")
    finally:
        train_batch_iterator.close()

    if config.is_distributed:
        dist.barrier()

    for callback in callbacks:
        callback.post_train_loop()

    if config.is_local_main_process:
        accelerator.save_complete_weights_from_checkpoint(
            config.best_state_path, config.final_weights_path
        )

    if not config.is_distributed:
        return accelerator.model
    else:
        return None


def _cycle_through_epochs(dataloader: DataLoader, is_distributed: bool):
    epoch = 0
    while True:
        if is_distributed and isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(epoch)
        for batch in dataloader:
            yield batch
        epoch += 1


def _check_dataset(dataset, split: str):
    try:
        len(dataset)
    except TypeError:
        if not isinstance(dataset, IterableDataset):
            warnings.warn(
                f"Dataset for {split} split appears to be a streaming/iterable dataset, "
                "but is not an instance of 'torch.utils.data.IterableDataset'. This could cause issues "
                "within the DataLoader.",
                UserWarning,
            )


def _check_dataloader(dataloader: DataLoader):
    # If using a regular dataset and not streaming/iterable dataset, we
    # should probably be using a `DistributedSampler`.
    if not isinstance(dataloader.dataset, IterableDataset) and not isinstance(
        dataloader.sampler, DistributedSampler
    ):
        warnings.warn(
            "DistributedSampler is required for dataloader during distributed training, "
            f"found {type(dataloader.sampler)} instead.",
            UserWarning,
        )
