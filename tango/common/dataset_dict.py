from dataclasses import dataclass, field
from typing import Mapping, Any, Sequence, TypeVar, Generic


T = TypeVar("T")


@dataclass
class DatasetDict(Generic[T]):
    """
    This definition of a dataset combines all splits and arbitrary metadata into
    one handy class.
    """

    splits: Mapping[str, Sequence[T]]
    """
    Maps the name of the split to a sequence of instances of type ``T``.
    """

    metadata: Mapping[str, Any] = field(default_factory=dict)
    """Metadata can contain anything you need."""

    def __getitem__(self, split: str) -> Sequence[T]:
        """
        Get a split.
        """
        return self.splits[split]

    def __len__(self) -> int:
        """
        Returns the number of splits.
        """
        return len(self.splits)

    def keys(self):
        return self.splits.keys()
