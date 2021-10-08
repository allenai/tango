from itertools import islice
import os
import random
import tempfile
import typing as t

from more_itertools import chunked
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DistributedSampler

from .data import DataLoader
from .format import TorchFormat
from .model import Model
from .optim import Optimizer, LRScheduler
from tango.common.dataset_dict import DatasetDict
from tango.common.lazy import Lazy
from tango.common.tqdm import Tqdm
from tango.format import Format
from tango.step import Step


@Step.register("torch::train")
class TorchTrainStep(Step):
    """
    A basic training loop step.

    .. tip::

        Registered as a :class:`~tango.step.Step` under the name ``torch::train``.
    """

    DETERMINISTIC: bool = True
    CACHEABLE = True
    FORMAT: Format = TorchFormat()

    def run(  # type: ignore[override]
        self,
        model: Model,
        dataset_dict: DatasetDict,
        train_dataloader: Lazy[DataLoader],
        optimizer: Lazy[Optimizer],
        *,
        train_split: str = "train",
        validation_split: t.Optional[str] = None,
        lr_scheduler: t.Optional[Lazy[LRScheduler]] = None,
        validation_dataloader: t.Optional[Lazy[DataLoader]] = None,
        seed: int = 42,
        train_steps: t.Optional[int] = None,
        validation_steps: t.Optional[int] = None,
        grad_accum: int = 1,
        log_every: int = 10,
        checkpoint_every: int = 100,
        validate_every: int = 100,
    ) -> Model:
        """
        Run a basic training loop to train the ``model``.

        Parameters
        ----------

        model :
            The model to train. It should return a ``dict`` that includes the ``loss``
            during training and validation.
        dataset_dict :
            The train and optional validation data.
        train_dataloader :
            The data loader that generates training batches. The batches should be :class:`dict`
            objects.
        optimizer :
            A PyTorch :class:`~torch.optim.Optimizer`.
        train_split :
            The name of the data split used for training in the ``dataset_dict``.
        validation_split :
            Optional name of the validation split in the ``dataset_dict``.
        lr_scheduler :
            An optional
            `learning rate scheduler <https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate>`_
            to adjust the learning rate throughout training.
        validation_dataloader :
            An optional data loader for generating validation batches. The batches should be
            :class:`dict` objects. If not specified, but ``validation_split`` is given,
            the validation ``DataLoader`` will be constructed from the same parameters
            as the train ``DataLoader``.
        seed :
            Used to set the RNG states at the beginning of training.
        train_steps :
            The number of steps to train for. If not specified training will
            stop after a complete iteration through the `train_dataloader`.
        validation_steps :
            The number of steps to validate for. If not specified validation
            will stop after a complete iteration through the `validation_dataloader`.
        grad_accum :
            The number of gradient accumulation steps.
        log_every :
            Log every this many steps.
        checkpoint_every :
            Save a checkpoint every this many steps.
        validate_every :
            Run the validation loop every this many steps.

        """
        device: torch.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.logger.info("CUDA is available")
            device = torch.device("cuda")

        # TODO
        is_distributed = False
        worker_id = 0
        is_local_main_process = worker_id == 0

        state_path = self.work_dir / f"state_worker{worker_id}.pt"

        # Construct data loaders.
        if validation_split is not None:
            if validation_dataloader is not None:
                validation_dataloader = validation_dataloader.construct(
                    dataset=dataset_dict[validation_split]
                )
            else:
                validation_dataloader = train_dataloader.construct(
                    dataset=dataset_dict[validation_split]
                )
        validation_dataloader: t.Optional[DataLoader] = t.cast(
            t.Optional[DataLoader], validation_dataloader
        )
        train_dataloader: DataLoader = train_dataloader.construct(dataset=dataset_dict[train_split])

        if train_steps is None:
            train_steps = len(train_dataloader)
        train_steps: int = t.cast(int, train_steps)  # type: ignore[no-redef]
        if validation_dataloader is not None:
            if validation_steps is None:
                validation_steps = len(validation_dataloader)

        # Check working directory to see if we should recover from a previous run.
        initial_state: t.Optional[t.Dict[str, t.Any]] = None
        if state_path.is_file():
            self.logger.info("Recovering from previous run")
            initial_state = torch.load(state_path)

        # Prepare model.
        model = model.to(device)
        if initial_state is not None:
            model.load_state_dict(initial_state["model"])

        # Prepare optimizer and lr scheduler.
        optimizer: Optimizer = optimizer.construct(params=model.parameters())
        if initial_state is not None:
            optimizer.load_state_dict(initial_state["optimizer"])
        if lr_scheduler is not None:
            lr_scheduler = lr_scheduler.construct(optimizer=optimizer)
            if initial_state is not None:
                lr_scheduler.load_state_dict(initial_state["lr_scheduler"])
        lr_scheduler: t.Optional[LRScheduler] = t.cast(t.Optional[LRScheduler], lr_scheduler)

        # Set random seeds.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        model.train()
        val_loss: t.Optional[float] = None
        training_batches = enumerate(
            islice(
                chunked(cycle_through_epochs(train_dataloader, is_distributed), grad_accum),
                train_steps,
            )
        )

        # Catch data loader up to where we left off before.
        if initial_state is not None:
            training_steps = initial_state["training_steps"]
            self.logger.info("Catching data loader up to step %d", training_steps)
            for step, batch in training_batches:
                del batch
                if step >= training_steps - 1:
                    break

        with Tqdm.tqdm(
            training_batches,
            desc="Training",
            initial=0 if initial_state is None else initial_state["training_steps"],
            total=train_steps,
            disable=not is_local_main_process,
        ) as train_batch_iterator:
            for step, batch in train_batch_iterator:

                def save_state():
                    temp_state_file = tempfile.NamedTemporaryFile(
                        "w+b", dir=self.work_dir, delete=False, suffix="pt"
                    )
                    try:
                        torch.save(
                            {
                                "optimizer": optimizer.state_dict(),
                                "scheduler": None
                                if lr_scheduler is None
                                else lr_scheduler.state_dict(),
                                "model": model.state_dict(),
                                "training_steps": step + 1,
                            },
                            temp_state_file.name,
                        )
                        temp_state_file.close()
                        os.replace(temp_state_file.name, state_path)
                    finally:
                        if os.path.exists(temp_state_file.name):
                            os.remove(temp_state_file.name)

                optimizer.zero_grad()
                batch_loss = 0.0
                for micro_batch in batch:
                    # Move tensors to right device.
                    micro_batch = move_to_device(micro_batch, device)

                    # Get loss.
                    outputs = model(**micro_batch)
                    micro_batch_loss = outputs["loss"] / len(batch)
                    batch_loss += micro_batch_loss.detach().item()

                    # Calculate gradients.
                    micro_batch_loss.backward()

                    # Clean up in case it saves memory.
                    del micro_batch
                    del outputs
                    del micro_batch_loss

                del batch

                # Take step.
                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()

                should_log_this_step = step % log_every == 0 or step == train_steps - 1
                should_checkpoint_this_step = step > 0 and step % checkpoint_every == 0
                should_validate_this_step = validation_dataloader is not None and (
                    (step > 0 and step % validate_every == 0) or step == train_steps - 1
                )

                # Gather average loss across all workers.
                if should_log_this_step or should_validate_this_step and is_distributed:
                    # TODO: gather batch_loss across workers in distributed case.
                    pass

                # Update progress bar.
                if is_local_main_process and should_log_this_step:
                    if val_loss is not None:
                        train_batch_iterator.set_postfix(
                            batch_loss=batch_loss,
                            val_loss=val_loss,
                        )
                    else:
                        train_batch_iterator.set_postfix(batch_loss=batch_loss)

                # Checkpoint.
                if should_checkpoint_this_step:
                    save_state()

                # Validate.
                if should_validate_this_step:
                    assert validation_dataloader is not None
                    assert validation_steps is not None

                    # Prepare model for validation.
                    model.eval()
                    optimizer.zero_grad()  # Not strictly necessary.

                    running_loss = 0.0
                    with Tqdm.tqdm(
                        islice(validation_dataloader, validation_steps),
                        desc="Validating",
                        total=validation_steps,
                        leave=False,
                        disable=not is_local_main_process,
                    ) as val_batch_iterator:
                        for val_step, val_batch in enumerate(val_batch_iterator):
                            # Move tensors to right device.
                            val_batch = move_to_device(val_batch, device)

                            # Get loss.
                            with torch.inference_mode():
                                outputs = model(**val_batch)
                            loss = outputs["loss"]

                            running_loss += loss.item()
                            val_loss = running_loss / (val_step + 1)

                            # Update progress bar.
                            if is_local_main_process and val_step % 10 == 0:
                                val_batch_iterator.set_postfix(loss=val_loss)

                            # Clean up.
                            del val_batch
                            del outputs
                            del loss

                    # Average loss across all workers.
                    if is_distributed:
                        # TODO: gather loss across workers.
                        pass

                    # Reset model to train mode.
                    model.train()

                    # Update progress bar again.
                    if is_local_main_process:
                        train_batch_iterator.set_postfix(
                            batch_loss=batch_loss,
                            val_loss=val_loss,
                        )

        return model


def cycle_through_epochs(dataloader: DataLoader, is_distributed: bool):
    epoch = 0
    while True:
        if is_distributed and isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(epoch)
        for batch in dataloader:
            yield batch
        epoch += 1


T = t.TypeVar("T")


def move_to_device(o: T, device: torch.device) -> T:
    if isinstance(o, Tensor):
        return o.to(device)
    elif isinstance(o, dict):
        return {k: move_to_device(v, device) for k, v in o.items()}  # type: ignore[return-value]
    elif isinstance(o, list):
        return [move_to_device(x, device) for x in o]  # type: ignore[return-value]
    elif isinstance(o, tuple):
        return tuple((move_to_device(x, device) for x in o))  # type: ignore[return-value]
    else:
        return o
