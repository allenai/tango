from typing import Any, Dict, Generic, List, Optional, TypeVar, Union

import torch

from tango.common.lazy import Lazy
from tango.common.registrable import Registrable

T = TypeVar("T")


class DataCollator(Generic[T], Registrable):
    """
    A :class:`~tango.common.Registrable` version of a ``collate_fn``
    for a ``DataLoader``.

    Subclasses just need to implement :meth:`__call__()`.
    """

    default_implementation = "concat_tensor_dicts"
    """
    The default implementation is :class:`ConcatTensorDictsCollator`.
    """

    def __call__(self, items: List[T]) -> Dict[str, Any]:
        """
        Takes a list of items from a dataset and combines them into a batch.
        """
        raise NotADirectoryError


@DataCollator.register("concat_tensor_dicts")
class ConcatTensorDictsCollator(DataCollator[Dict[str, Any]]):
    """
    A simple ``collate_fn`` that expects items to be dictionaries of tensors.
    The tensors are just concatenated together.

    .. tip::

        Registered as a :class:`DataCollator` under the name "concat_tensor_dicts".
    """

    def __call__(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        out = {}
        keys = items[0].keys()
        for key in keys:
            if isinstance(items[0][key], torch.Tensor):
                out[key] = torch.cat([item[key].unsqueeze(0) for item in items])
            else:
                out[key] = [item[key] for item in items]  # type: ignore[assignment]
        return out


class Sampler(torch.utils.data.Sampler, Registrable):
    """
    A :class:`~tango.common.Registrable` version of a PyTorch
    :class:`~torch.utils.data.Sampler`.

    All `built-in PyTorch samplers
    <https://pytorch.org/docs/stable/data.html#data-loading-order-and-sampler>`_
    are registered under their corresponding class name (e.g. "RandomSampler").
    """


@Sampler.register("BatchSampler")
class BatchSampler(torch.utils.data.BatchSampler, Sampler):
    def __init__(self, sampler: Sampler, batch_size: int, drop_last: bool) -> None:
        super().__init__(sampler, batch_size, drop_last)


# Register all remaining samplers.
for name, cls in torch.utils.data.__dict__.items():
    if (
        isinstance(cls, type)
        and issubclass(cls, torch.utils.data.Sampler)
        and not cls == torch.utils.data.Sampler
        and name not in Sampler.list_available()
    ):
        Sampler.register("torch::" + name)(cls)


class DataLoader(torch.utils.data.DataLoader, Registrable):
    """
    A :class:`~tango.common.Registrable` version of a PyTorch
    :class:`~torch.utils.data.DataLoader`.
    """

    default_implementation = "default"

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        collate_fn: Optional[DataCollator] = ConcatTensorDictsCollator(),
        sampler: Optional[Union[Lazy[Sampler], Sampler]] = None,
        **kwargs
    ):
        super().__init__(
            dataset,
            collate_fn=collate_fn,
            sampler=sampler.construct(data_source=dataset, dataset=dataset)
            if isinstance(sampler, Lazy)
            else sampler,
            **kwargs
        )


DataLoader.register("default")(DataLoader)
