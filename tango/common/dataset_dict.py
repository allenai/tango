from dataclasses import dataclass, field
from typing import Mapping, Any, Sequence, TypeVar, Generic, Optional

from .det_hash import CustomDetHash


T = TypeVar("T")


@dataclass
class DatasetDict(CustomDetHash, Generic[T]):
    """
    This definition of a dataset combines all splits and arbitrary metadata into
    one handy class.
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

    When this is specified it will be used by the step caching mechanism to
    determine when the data has changed. Otherwise the caching mechanism will
    have to fall back to pickling the whole dataset dict to calculate a hash,
    which can be slow, so it's recommend you set this whenever possible.
    """

    def det_hash_object(self) -> Any:
        if self.fingerprint is not None:
            return self.fingerprint
        else:
            return self

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
