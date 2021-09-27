from tempfile import TemporaryDirectory
from typing import Optional, Iterable

import pytest

from tango.format import _OPEN_FUNCTIONS, DillFormat, JsonFormat


@pytest.mark.parametrize("compress", _OPEN_FUNCTIONS.keys())
def test_iterable_dill_format(compress: Optional[str]):
    r = (x + 1 for x in range(10))

    with TemporaryDirectory(prefix="test_iterable_dill_format-") as d:
        format = DillFormat[Iterable[int]](compress)
        format.write(r, d)
        r2 = format.read(d)
        assert [x + 1 for x in range(10)] == list(r2)


@pytest.mark.parametrize("compress", _OPEN_FUNCTIONS.keys())
def test_iterable_json_format(compress: Optional[str]):
    r = (x + 1 for x in range(10))

    with TemporaryDirectory(prefix="test_iterable_json_format-") as d:
        format = JsonFormat[Iterable[int]](compress)
        format.write(r, d)
        r2 = format.read(d)
        assert [x + 1 for x in range(10)] == list(r2)
