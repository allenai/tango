import os
import shutil
from os import PathLike
from typing import Any, Iterable, MutableSequence, Union

from sqlitedict import SqliteDict

from tango.common.sequences import SlicedSequence


class SqliteSparseSequence(MutableSequence[Any]):
    """
    This is a sparse sequence that pickles elements to a Sqlite database.

    When you read from the sequence, elements are retrieved and unpickled lazily. That means creating/opening
    a sequence is very fast and does not depend on the length of the sequence.

    This is a "sparse sequence" because you can set element ``n`` before you set element ``n-1``:

    .. testcode::
        :hide:

        from tango.common.sqlite_sparse_sequence import SqliteSparseSequence
        import tempfile
        dir = tempfile.TemporaryDirectory()
        from pathlib import Path
        filename = Path(dir.name) / "test.sqlite"

    .. testcode::

        s = SqliteSparseSequence(filename)
        element = "Big number, small database."
        s[2**32] = element
        assert len(s) == 2**32 + 1
        assert s[2**32] == element
        assert s[1000] is None
        s.close()

    .. testcode::
        :hide:

        dir.cleanup()

    You can use a ``SqliteSparseSequence`` from multiple processes at the same time. This is useful, for example,
    if you're filling out a sequence and you are partitioning ranges to processes.

    :param filename: the filename at which to store the data
    :param read_only: Set this to ``True`` if you only want to read.
    """

    def __init__(self, filename: Union[str, PathLike], read_only: bool = False):
        self.table = SqliteDict(filename, "sparse_sequence", flag="r" if read_only else "c")

    def __del__(self):
        self.close()

    def __getitem__(self, i: Union[int, slice]) -> Any:
        if isinstance(i, int):
            try:
                return self.table[str(i)]
            except KeyError:
                current_length = len(self)
                if i >= current_length or current_length <= 0:
                    raise IndexError("list index out of range")
                elif i < 0 < current_length:
                    return self.__getitem__(i % current_length)
                else:
                    return None
        elif isinstance(i, slice):
            return SlicedSequence(self, i)
        else:
            raise TypeError(f"list indices must be integers or slices, not {i.__class__.__name__}")

    def __setitem__(self, i: Union[int, slice], value: Any):
        if isinstance(i, int):
            current_length = len(self)
            if i < 0:
                i %= current_length
            self.table[str(i)] = value
            self.table["_len"] = max(i + 1, current_length)
            self.table.commit()
        else:
            raise TypeError(f"list indices must be integers, not {i.__class__.__name__}")

    def __delitem__(self, i: Union[int, slice]):
        current_length = len(self)
        if isinstance(i, int):
            if i < 0:
                i %= current_length
            if i >= current_length:
                raise IndexError("list assignment index out of range")
            for index in range(i + 1, current_length):
                self.table[str(index - 1)] = self.table.get(str(index))
            del self.table[str(current_length - 1)]
            self.table["_len"] = current_length - 1
            self.table.commit()
        elif isinstance(i, slice):
            # This isn't very efficient for continuous slices.
            for index in reversed(range(*i.indices(current_length))):
                del self[index]
        else:
            raise TypeError(f"list indices must be integers or slices, not {i.__class__.__name__}")

    def extend(self, values: Iterable[Any]) -> None:
        current_length = len(self)
        index = -1
        for index, value in enumerate(values):
            self.table[str(index + current_length)] = value
        if index < 0:
            return
        self.table["_len"] = current_length + index + 1
        self.table.commit()

    def insert(self, i: int, value: Any) -> None:
        current_length = len(self)
        for index in reversed(range(i, current_length)):
            self.table[str(index + 1)] = self.table.get(str(index))
        self.table[str(i)] = value
        self.table["_len"] = max(i + 1, current_length + 1)
        self.table.commit()

    def __len__(self) -> int:
        try:
            return self.table["_len"]
        except KeyError:
            return 0

    def clear(self) -> None:
        """
        Clears the entire sequence
        """
        self.table.clear()
        self.table.commit()

    def close(self) -> None:
        """
        Closes the underlying Sqlite table. Do not use this sequence afterwards!
        """
        if self.table is not None:
            self.table.close()
            self.table = None

    def copy_to(self, target: Union[str, PathLike]):
        """
        Make a copy of this sequence at a new location.

        :param target: the location of the copy

        This will attempt to make a hardlink, which is very fast, but only works on Linux and if ``target`` is
        on the same drive. If making a hardlink fails, it falls back to making a regular copy. As a result,
        there is no guarantee whether you will get a hardlink or a copy. If you get a hardlink, future edits
        in the source sequence will also appear in the target sequence. This is why we recommend to not use
        :meth:`copy_to()` until you are done with the sequence. This is not ideal, but it is a compromise we make
        for performance.
        """
        try:
            os.link(self.table.filename, target)
        except OSError as e:
            if e.errno == 18:  # Cross-device link
                shutil.copy(self.table.filename, target)
            else:
                raise
