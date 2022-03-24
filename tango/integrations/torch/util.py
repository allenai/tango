import random
import warnings
from collections import UserDict
from typing import Dict, Optional, TypeVar, Union

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler, IterableDataset

from .data import DataLoader

T = TypeVar("T")


def move_to_device(o: T, device: torch.device) -> T:
    if isinstance(o, torch.Tensor):
        return o.to(device)  # type: ignore[return-value]
    elif isinstance(o, dict) or isinstance(o, UserDict):
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


def peak_gpu_memory(reset: bool = False) -> Dict[int, int]:
    """
    Get the peak GPU memory usage in MiB by distributed worker rank.

    :returns:
        Keys are rank ids as integers (from 0 up to world size - 1).
        Values are memory usage as integers in MiB.
        Returns an empty `dict` if GPUs are not available.
    """
    if not torch.cuda.is_available():
        return {}

    device = torch.device("cuda")

    results_dict: Dict[int, int] = {}
    if dist.is_available() and dist.is_initialized():
        # If the backend is not 'nccl', we're training on CPU.
        if dist.get_backend() != "nccl":
            return {}

        global_rank = dist.get_rank()
        world_size = dist.get_world_size()
        peak_mb = torch.cuda.max_memory_allocated(device) // 1048576
        peak_mb_tensor = torch.tensor([global_rank, peak_mb], device=device)
        # All of these tensors will be gathered into this list.
        gather_results = [torch.tensor([0, 0], device=device) for _ in range(world_size)]

        dist.all_gather(gather_results, peak_mb_tensor)

        for peak_mb_tensor in gather_results:
            results_dict[int(peak_mb_tensor[0])] = int(peak_mb_tensor[1])
    else:
        results_dict = {0: torch.cuda.max_memory_allocated()}

    if reset:
        # Reset peak stats.
        torch.cuda.reset_max_memory_allocated(device)

    return results_dict
