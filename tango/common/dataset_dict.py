from dataclasses import dataclass, field
from typing import Any, Generic, Iterable, Iterator, Mapping, Sequence, TypeVar

T = TypeVar("T")
S = TypeVar("S")


@dataclass
class DatasetDictBase(Generic[S], Mapping[str, S]):
    """
    The base class for :class:`DatasetDict` and :class:`IterableDatasetDict`.
    """

    splits: Mapping[str, S]
    """
    A mapping of dataset split names to splits.
    """

    metadata: Mapping[str, Any] = field(default_factory=dict)
    """
    Metadata can contain anything you need.
    """

    def __getitem__(self, split: str) -> S:
        """
        Get a split in :attr:`splits`.
        """
        return self.splits[split]

    def __contains__(self, split: str) -> bool:  # type: ignore[override]
        """
        Checks if :attr:`splits` contains the given split.
        """
        return split in self.splits

    def __iter__(self) -> Iterator[str]:
        """
        Returns an iterator over the keys in :attr:`splits`.
        """
        return iter(self.splits.keys())

    def __len__(self) -> int:
        """
        Returns the number of splits in :attr:`splits`.
        """
        return len(self.splits)

    def keys(self):
        """
        Returns the split names in :attr:`splits`.
        """
        return self.splits.keys()


@dataclass
class DatasetDict(DatasetDictBase[Sequence[T]], Generic[T]):
    """
    A generic :class:`~collections.abc.Mapping` class of split names (:class:`str`) to datasets
    (``Sequence[T]``).
    """


@dataclass
class IterableDatasetDict(DatasetDictBase[Iterable[T]], Generic[T]):
    """
    An "iterable" version of :class:`DatasetDict`, where the dataset splits have
    type ``Iterable[T]`` instead of ``Sequence[T]``. This is useful for streaming datasets.
    """
