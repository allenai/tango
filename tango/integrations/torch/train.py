import os
import random
import tempfile
import warnings
from itertools import islice
from typing import Any, Dict, List, Optional, Set, TypeVar, cast

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from more_itertools import chunked
from torch import Tensor
from torch.utils.data import DistributedSampler, IterableDataset

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
        train_epochs: Optional[int] = None,
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
        train_epochs : :class:`int`, optionsl
            The number of epochs to train for. You cannot specify `train_steps` and `train_epochs`
            at the same time.
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
        amp : :class:`bool`, optional
            Use automatic mixed precision. Default is ``False``.
        max_grad_norm : :class:`float`, optional
            If set, gradients will be clipped to have this max norm. Default is ``None``.
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
            amp=amp,
            max_grad_norm=max_grad_norm,
            is_distributed=is_distributed,
            devices=devices,
            distributed_port=distributed_port,
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
                    get_extra_imported_modules(),
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
            )
            assert final_model is not None
            final_model = final_model.cpu()

        # Load best checkpoint before returning model.
        best_state_path = self.work_dir / "state_worker0_best.pt"
        if best_state_path.is_file():
            print(f"Loading best weights from {best_state_path.resolve().name}")
            state = torch.load(best_state_path, map_location="cpu")
            final_model.load_state_dict(state["model"])

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

        common_logging.initialize_logging(prefix=f"[worker {worker_id}]")

    # Resolve and set device.
    device = config.worker_local_default_device
    if config.is_distributed and device != torch.device("cpu"):
        torch.cuda.set_device(device)

    # Init distributed process group.
    if config.is_distributed:
        backend = "gloo" if device == torch.device("cpu") else "nccl"
        dist.init_process_group(
            backend=backend,
            init_method=f"tcp://127.0.0.1:{config.distributed_port}",
            world_size=config.world_size,
            rank=worker_id,
        )

    grad_scaler: Optional[torch.cuda.amp.GradScaler] = None
    if config.amp:
        grad_scaler = torch.cuda.amp.GradScaler()

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
        _check_dataloader(train_dataloader)
        if validation_dataloader is not None:
            _check_dataloader(validation_dataloader)

    # Check working directory to see if we should recover from a previous run.
    initial_state: Optional[Dict[str, Any]] = None
    if config.state_path.is_file():
        if config.is_local_main_process:
            print(f"Recovering from previous run at {str(config.state_path.name)}")
        initial_state = torch.load(config.state_path)

    # Prepare model.
    model: Model = model.construct()
    if initial_state is not None:
        model.load_state_dict(initial_state["model"])
    model = model.to(device)
    if config.is_distributed:
        model = cast(Model, nn.parallel.DistributedDataParallel(model))

    # Prepare optimizer and lr scheduler.
    optimizer: Optimizer = optimizer.construct(params=model.parameters())
    if initial_state is not None:
        optimizer.load_state_dict(initial_state["optimizer"])
    lr_scheduler_: Optional[LRScheduler] = None
    if lr_scheduler is not None:
        lr_scheduler_ = lr_scheduler.construct(optimizer=optimizer)
        if initial_state is not None:
            lr_scheduler_.load_state_dict(initial_state["scheduler"])
    lr_scheduler: Optional[LRScheduler] = lr_scheduler_

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
            model=model,
            optimizer=optimizer,
            dataset_dict=dataset_dict,
            train_dataloader=train_dataloader,
            validation_dataloader=validation_dataloader,
            lr_scheduler=lr_scheduler,
        )
        for callback in (callbacks or [])
    ]
    if initial_state:
        for callback, state in zip(callbacks, initial_state["callbacks"]):
            callback.load_state_dict(state)

    del initial_state

    model.train()
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
        temp_state_file = tempfile.NamedTemporaryFile(
            "w+b", dir=config.work_dir, delete=False, suffix=".pt"
        )
        try:
            with Tqdm.wrapattr(
                temp_state_file,
                "write",
                desc="Saving checkpoint",
                leave=False,
                disable=not config.is_local_main_process,
            ) as f:
                checkpoint_state = {
                    "optimizer": optimizer.state_dict(),  # type: ignore[attr-defined]
                    "scheduler": None
                    if lr_scheduler is None
                    else lr_scheduler.state_dict(),  # type: ignore[attr-defined]
                    "model": model.module.state_dict()  # type: ignore[attr-defined]
                    if config.is_distributed
                    else model.state_dict(),  # type: ignore[attr-defined]
                    "training_steps": step + 1,
                    "best_batch_loss": best_batch_loss,
                    f"val_{config.val_metric_name}": val_metric,
                    f"best_{config.val_metric_name}": best_val_metric,
                    "callbacks": [
                        callback.state_dict() for callback in callbacks  # type: ignore[union-attr]
                    ],
                }
                for callback in callbacks:  # type: ignore[union-attr]
                    callback.pre_checkpoint(checkpoint_state)  # type: ignore[union-attr]
                torch.save(checkpoint_state, f)
            temp_state_file.close()
            os.replace(temp_state_file.name, config.state_path_for_step(step))

            # Link to most recent state path.
            if config.state_path.is_file():
                config.state_path.unlink()
            os.link(config.state_path_for_step(step), config.state_path)

            # Link to best state path.
            if is_best_checkpoint():
                if config.best_state_path.is_file():
                    config.best_state_path.unlink()
                os.link(config.state_path_for_step(step), config.best_state_path)

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

            for callback in callbacks:  # type: ignore[union-attr]
                callback.post_checkpoint(config.state_path_for_step)  # type: ignore[union-attr]
        finally:
            if os.path.exists(temp_state_file.name):
                os.remove(temp_state_file.name)

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

    train_batch_iterator = Tqdm.tqdm(
        training_batches,
        desc="Training",
        initial=start_step,
        total=config.train_steps,
        disable=not config.is_local_main_process,
    )
    try:
        for step, (epoch, batch) in train_batch_iterator:
            if epoch != current_epoch:
                # Start of new epoch.
                if epoch > 0:
                    # Call post-epoch callbacks for the last epoch.
                    for callback in callbacks:
                        callback.post_epoch(current_epoch)
                for callback in callbacks:
                    callback.pre_epoch(epoch)
                current_epoch = epoch

            # Pre-batch callback.
            for callback in callbacks:
                callback.pre_batch(step, batch)

            optimizer.zero_grad()
            batch_loss = 0.0
            for micro_batch in batch:
                # Move tensors to right device.
                micro_batch = _move_to_device(micro_batch, device)

                # Get loss.
                with torch.cuda.amp.autocast(enabled=config.amp):
                    outputs = model(**micro_batch)
                    micro_batch_loss = outputs["loss"] / len(batch)

                if torch.isnan(micro_batch_loss):
                    raise ValueError("nan loss encountered")
                batch_loss += micro_batch_loss.detach().item()

                # Calculate gradients.
                if grad_scaler is not None:
                    grad_scaler.scale(micro_batch_loss).backward()
                else:
                    micro_batch_loss.backward()

                # Clean up in case it saves memory.
                del micro_batch
                del outputs
                del micro_batch_loss

            # Post-batch callback.
            for callback in callbacks:
                callback.post_batch(step, batch_loss)

            if best_batch_loss is None or batch_loss <= best_batch_loss:
                best_batch_loss = batch_loss

            del batch

            # Unscale gradients.
            if grad_scaler is not None:
                grad_scaler.unscale_(optimizer)

            # Clip gradients.
            if config.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            # Take optimizer step.
            if grad_scaler is not None:
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                optimizer.step()

            # Adjust LR schedule.
            if lr_scheduler is not None:
                lr_scheduler.step()

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
                model.eval()
                optimizer.zero_grad()  # Not strictly necessary.

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
                        # Move tensors to right device.
                        val_batch = _move_to_device(val_batch, device)

                        # Get metric.
                        with torch.cuda.amp.autocast(enabled=config.amp):
                            with torch.inference_mode():
                                outputs = model(**val_batch)
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
                model.train()

                if best_val_metric is None:
                    best_val_metric = val_metric
                elif config.minimize_val_metric and val_metric <= best_val_metric:
                    best_val_metric = val_metric
                elif not config.minimize_val_metric and val_metric >= best_val_metric:
                    best_val_metric = val_metric

                # Post validation callback.
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

        # End train loop

        # Final post-epoch callback.
        for callback in callbacks:
            callback.post_epoch(current_epoch)
    except StopEarly:
        if config.is_local_main_process:
            print("Stopping early!")
    finally:
        train_batch_iterator.close()

    if config.is_distributed:
        dist.barrier()

    for callback in callbacks:
        callback.post_train_loop()

    if not config.is_distributed:
        return model
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


T = TypeVar("T")


def _move_to_device(o: T, device: torch.device) -> T:
    if isinstance(o, Tensor):
        return o.to(device)  # type: ignore[return-value]
    elif isinstance(o, dict):
        return {k: _move_to_device(v, device) for k, v in o.items()}  # type: ignore[return-value]
    elif isinstance(o, list):
        return [_move_to_device(x, device) for x in o]  # type: ignore[return-value]
    elif isinstance(o, tuple):
        return tuple((_move_to_device(x, device) for x in o))  # type: ignore[return-value]
    else:
        return o


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
