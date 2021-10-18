from dataclasses import dataclass, field
from typing import Mapping, Any, Sequence, TypeVar, Generic, Optional, Iterator

from .det_hash import CustomDetHash


T = TypeVar("T")


@dataclass
class DatasetDict(CustomDetHash, Generic[T], Mapping[str, Sequence[T]]):
    """
    A generic :class:`~collections.abc.Mapping` class of split names (:class:`str`) to datasets
    (``Sequence[T]``).
    """

    splits: Mapping[str, Sequence[T]]
    """
    Maps the name of the split to a sequence of instances of type ``T``.
    """

    metadata: Mapping[str, Any] = field(default_factory=dict)
    """
    Metadata can contain anything you need.
    """

    fingerprint: Optional[str] = None
    """
    A unique fingerprint associated with the data.

    .. important::
        When this is specified it will be used by the step caching mechanism to
        determine when the data has changed. Otherwise the caching mechanism will
        have to fall back to pickling the whole dataset dict to calculate a hash,
        which can be slow, so it's recommend you set this whenever possible.
    """

    def det_hash_object(self) -> Any:
        """
        Overrides :meth:`~tango.common.det_hash.CustomDetHash.det_hash_object` to return
        :attr:`fingerprint`  when specified instead of ``self``.
        """
        if self.fingerprint is not None:
            return self.fingerprint
        else:
            return self

    def __getitem__(self, split: str) -> Sequence[T]:
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
