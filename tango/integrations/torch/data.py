import typing as t
import torch

from tango.common.lazy import Lazy
from tango.common.registrable import Registrable


T = t.TypeVar("T")


class DataCollator(t.Generic[T], Registrable):
    """
    A :class:`~tango.common.registrable.Registrable` version of a ``collate_fn``
    for a ``DataLoader``.

    Subclasses just need to implement :meth:`__call__()`.
    """

    default_implementation = "concat_tensor_dicts"
    """
    The default implementation is :class:`ConcatTensorDictsCollator`.
    """

    def __call__(self, items: t.List[T]) -> t.Dict[str, t.Any]:
        """
        Takes a list of items from a dataset and combines them into a batch.
        """
        raise NotADirectoryError


@DataCollator.register("concat_tensor_dicts")
class ConcatTensorDictsCollator(DataCollator[t.Dict[str, torch.Tensor]]):
    """
    A simple ``collate_fn`` that expects items to be dictionaries of tensors.
    The tensors are just concatenated together.

    .. tip::

        Registered as a :class:`DataCollator` under the name "concat_tensor_dicts".
    """

    def __call__(self, items: t.List[t.Dict[str, torch.Tensor]]) -> t.Dict[str, t.Any]:
        out = {}
        keys = items[0].keys()
        for key in keys:
            if isinstance(items[0][key], torch.Tensor):
                out[key] = torch.cat([item[key].unsqueeze(0) for item in items])
            else:
                out[key] = [item[key] for item in items]
        return out


class Sampler(torch.utils.data.Sampler, Registrable):
    """
    A :class:`~tango.common.registrable.Registrable` version of a PyTorch
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
        Sampler.register(name)(cls)


class DataLoader(torch.utils.data.DataLoader, Registrable):
    """
    A :class:`~tango.common.registrable.Registrable` version of a PyTorch
    :class:`~torch.utils.data.DataLoader`.
    """

    default_implementation = "default"

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        collate_fn: t.Optional[DataCollator] = ConcatTensorDictsCollator(),
        sampler: t.Optional[Lazy[Sampler]] = None,
        **kwargs
    ):
        if sampler is not None:
            sampler = sampler.construct(data_source=dataset, dataset=dataset)
        super().__init__(dataset, collate_fn=collate_fn, sampler=sampler, **kwargs)


DataLoader.register("default")(DataLoader)
