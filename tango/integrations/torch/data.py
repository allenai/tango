import typing as t
import torch

from tango.common.registrable import Registrable


T = t.TypeVar("T")


class DataCollator(t.Generic[T], Registrable):
    """
    A :class:`~tango.common.registrable.Registrable` version of a ``collate_fn``
    for a ``DataLoader``.
    """

    default_implementation = "concat_tensor_dicts"
    """
    The default implementation is :class:`ConcatTensorDictsCollator`.
    """

    def __call__(self, items: t.List[T]) -> t.Dict[str, t.Any]:
        raise NotADirectoryError


@DataCollator.register("concat_tensor_dicts")
class ConcatTensorDictsCollator(DataCollator[t.Dict[str, torch.Tensor]]):
    """
    A simple ``collate_fn`` that expects items to be dictionaries of tensors.
    The tensors are just concatenated together.

    .. tip::

        Registered as a :class:`DataCollator` under the name ``concat_tensor_dicts``.
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
        **kwargs
    ):
        super().__init__(dataset, collate_fn=collate_fn, **kwargs)


DataLoader.register("default")(DataLoader)
