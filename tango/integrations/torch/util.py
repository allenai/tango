import random
import warnings
from typing import Optional, TypeVar, Union

import numpy as np
import torch
from torch.utils.data import DistributedSampler, IterableDataset

from .data import DataLoader

T = TypeVar("T")


def move_to_device(o: T, device: torch.device) -> T:
    if isinstance(o, torch.Tensor):
        return o.to(device)  # type: ignore[return-value]
    elif isinstance(o, dict):
        return {k: move_to_device(v, device) for k, v in o.items()}  # type: ignore[return-value]
    elif isinstance(o, list):
        return [move_to_device(x, device) for x in o]  # type: ignore[return-value]
    elif isinstance(o, tuple):
        return tuple((move_to_device(x, device) for x in o))  # type: ignore[return-value]
    else:
        return o


def check_dataset(dataset, split: str):
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


def check_dataloader(dataloader: DataLoader):
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


def set_seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: Optional[Union[int, str, torch.device]] = None) -> torch.device:
    if device is None:
        if torch.cuda.is_available():
            # TODO (epwalsh, dirkgr): automatically pick which GPU to use when there are multiple
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    elif isinstance(device, int):
        if device >= 0:
            return torch.device(f"cuda:{device}")
        else:
            return torch.device("cpu")
    elif isinstance(device, str):
        return torch.device(device)
    elif isinstance(device, torch.device):
        return device
    else:
        raise TypeError(f"unexpected type for 'device': '{device}'")
