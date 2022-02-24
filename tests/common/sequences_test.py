import os
from tempfile import TemporaryDirectory

import pytest

from tango.common.sequences import (
    ConcatenatedSequence,
    MappedSequence,
    ShuffledSequence,
    SlicedSequence,
    SqliteSparseSequence,
)


def assert_equal_including_exceptions(expected_fn, actual_fn):
    try:
        expected = expected_fn()
    except Exception as e:
        with pytest.raises(e.__class__):
            actual_fn()
    else:
        assert expected == actual_fn()


def test_shuffled_sequence():
    seq = ShuffledSequence(list(range(10)))
    assert 5 in seq
    assert len(seq) == 10


def test_sliced_sequence():
    seq = SlicedSequence(list(range(10)), slice(10))
    assert len(seq) == 10
    assert seq[0] == 0
    assert seq[-1] == 9
    seq2 = seq[-2:]
    assert len(seq2) == 2


def test_concatenated_sequence():
    l1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    l2 = ConcatenatedSequence([0, 1], [], [2, 3, 4], [5, 6, 7, 8, 9], [])

    # __len__()
    assert len(l1) == len(l2)

    # index()
    for item in l1 + [999]:
        # no indices
        assert_equal_including_exceptions(lambda: l1.index(item), lambda: l2.index(item))

        # only start index
        for index in range(-15, 15):
            assert_equal_including_exceptions(
                lambda: l1.index(item, index), lambda: l2.index(item, index)
            )

        # start and stop index
        for start_index in range(-15, 15):
            for end_index in range(-15, 15):
                assert_equal_including_exceptions(
                    lambda: l1.index(item, start_index, end_index),
                    lambda: l2.index(item, start_index, end_index),
                )

    # __getitem__()
    for index in range(-15, 15):
        assert_equal_including_exceptions(lambda: l1[index], lambda: l2[index])

    for start_index in range(-15, 15):
        for end_index in range(-15, 15):
            assert_equal_including_exceptions(
                lambda: l1[start_index:end_index], lambda: list(l2[start_index:end_index])
            )

    # count()
    for item in l1 + [999]:
        assert_equal_including_exceptions(lambda: l1.count(item), lambda: l2.count(item))

    # __contains__()
    for item in l1 + [999]:
        assert_equal_including_exceptions(lambda: item in l1, lambda: item in l2)


def test_sqlite_sparse_sequence():
    with TemporaryDirectory(prefix="test_sparse_sequence-") as temp_dir:
        s = SqliteSparseSequence(os.path.join(temp_dir, "test.sqlite"))
        assert len(s) == 0
        s.extend([])
        assert len(s) == 0
        s.append("one")
        assert len(s) == 1
        s.extend(["two", "three"])
        s.insert(1, "two")
        assert s[1] == "two"
        assert s.count("two") == 2
        ss = s[1:3]
        assert list(ss) == ["two", "two"]
        del s[1:3]
        assert len(s) == 2
        assert s[-1] == "three"
        s.clear()
        assert len(s) == 0


def test_mapped_sequence():
    my_very_long_sequence = ["John", "Paul", "George", "Ringo"]
    m = MappedSequence(lambda x: len(x), my_very_long_sequence)
    assert m[0] == 4
    assert len(m) == len(my_very_long_sequence)
    for i in range(len(m)):
        assert m[i] == m[i:][0]
