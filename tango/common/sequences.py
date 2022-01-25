import bisect
import random
from collections import abc
from typing import Callable, Optional, Sequence, Union


class ShuffledSequence(abc.Sequence):
    """
    Produces a shuffled view of a sequence, such as a list.

    This assumes that the inner sequence never changes. If it does, the results
    are undefined.

    :param inner_sequence: the inner sequence that's being shuffled
    :param indices: Optionally, you can specify a list of indices here. If you don't, we'll just shuffle the
                    inner sequence randomly. If you do specify indices, element ``n`` of the output sequence
                    will be ``inner_sequence[indices[n]]``. This gives you great flexibility. You can repeat
                    elements, leave them out completely, or slice the list. A Python :class:`slice` object is
                    an acceptable input for this parameter, and so are other sequences from this module.

    Example:

    .. testcode::
        :hide:

        import random
        random.seed(42)

    .. testcode::

        from tango.common.sequences import ShuffledSequence
        l = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        shuffled_l = ShuffledSequence(l)

        print(shuffled_l[0])
        print(shuffled_l[1])
        print(shuffled_l[2])
        assert len(shuffled_l) == len(l)

    This will print something like the following:

    .. testoutput::

        4
        7
        8
    """

    def __init__(self, inner_sequence: Sequence, indices: Optional[Sequence[int]] = None):
        self.inner = inner_sequence
        self.indices: Sequence[int]
        if indices is None:
            self.indices = list(range(len(inner_sequence)))
            random.shuffle(self.indices)
        else:
            self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: Union[int, slice]):
        if isinstance(i, int):
            return self.inner[self.indices[i]]
        else:
            return ShuffledSequence(self.inner, self.indices[i])

    def __contains__(self, item) -> bool:
        for i in self.indices:
            if self.inner[i] == item:
                return True
        return False


class SlicedSequence(ShuffledSequence):
    """
    Produces a sequence that's a slice into another sequence, without copying the elements.

    This assumes that the inner sequence never changes. If it does, the results
    are undefined.

    :param inner_sequence: the inner sequence that's being shuffled
    :param s: the :class:`~slice` to slice the input with.

    Example:

    .. testcode::

        from tango.common.sequences import SlicedSequence
        l = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        sliced_l = SlicedSequence(l, slice(1, 4))

        print(sliced_l[0])
        print(sliced_l[1])
        print(sliced_l[2])
        assert len(sliced_l) == 3

    This will print the following:

    .. testoutput::

        2
        3
        4

    """

    def __init__(self, inner_sequence: Sequence, s: slice):
        super().__init__(inner_sequence, range(*s.indices(len(inner_sequence))))


class ConcatenatedSequence(abc.Sequence):
    """
    Produces a sequence that's the lazy concatenation of multiple other sequences. It does not copy
    any of the elements of the original sequences.

    This assumes that the inner sequences never change. If they do, the results are undefined.

    :param sequences: the inner sequences to concatenate

    Example:

    .. testcode::

        from tango.common.sequences import ConcatenatedSequence
        l1 = [1, 2, 3]
        l2 = [4, 5]
        l3 = [6]
        cat_l = ConcatenatedSequence(l1, l2, l3)

        assert len(cat_l) == 6
        for i in cat_l:
            print(i)

    This will print the following:

    .. testoutput::

        1
        2
        3
        4
        5
        6
    """

    def __init__(self, *sequences: Sequence):
        self.sequences = sequences
        self.cumulative_sequence_lengths = [0]
        for sequence in sequences:
            self.cumulative_sequence_lengths.append(
                self.cumulative_sequence_lengths[-1] + len(sequence)
            )

    def __len__(self):
        return self.cumulative_sequence_lengths[-1]

    def __getitem__(self, i: Union[int, slice]):
        if isinstance(i, int):
            if i < 0:
                i += len(self)
            if i < 0 or i >= len(self):
                raise IndexError("list index out of range")
            sequence_index = bisect.bisect_right(self.cumulative_sequence_lengths, i) - 1
            i -= self.cumulative_sequence_lengths[sequence_index]
            return self.sequences[sequence_index][i]
        else:
            return SlicedSequence(self, i)

    def __contains__(self, item) -> bool:
        return any(s.__contains__(item) for s in self.sequences)


class MappedSequence(abc.Sequence):
    """
    Produces a sequence that applies a function to every element of another sequence.

    This is similar to Python's :func:`map`, but it returns a sequence instead of a :class:`map` object.

    :param fn: the function to apply to every element of the inner sequence. The function should take
               one argument.
    :param inner_sequence: the inner sequence to map over

    Example:

    .. testcode::

        from tango.common.sequences import MappedSequence

        def square(x):
            return x * x

        l = [1, 2, 3, 4]
        map_l = MappedSequence(square, l)

        assert len(map_l) == len(l)
        for i in map_l:
            print(i)

    This will print the following:

    .. testoutput::

        1
        4
        9
        16

    """

    def __init__(self, fn: Callable, inner_sequence: Sequence):
        self.inner = inner_sequence
        self.fn = fn

    def __getitem__(self, item):
        item = self.inner.__getitem__(item)
        return self.fn(item)

    def __len__(self):
        return self.inner.__len__()

    def __contains__(self, item):
        return any(e == item for e in self)
