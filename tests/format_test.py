from typing import Dict, Iterable, Optional

import pytest

from tango.common.testing import TangoTestCase
from tango.format import _OPEN_FUNCTIONS, DillFormat, JsonFormat


class TestFormat(TangoTestCase):
    @pytest.mark.parametrize("compress", _OPEN_FUNCTIONS.keys())
    def test_dill_format(self, compress: Optional[str]):
        artifact = "Hello, World!"
        format = DillFormat[str](compress)
        format.write(artifact, self.TEST_DIR)
        assert format.read(self.TEST_DIR) == artifact

    @pytest.mark.parametrize("compress", _OPEN_FUNCTIONS.keys())
    def test_iterable_dill_format(self, compress: Optional[str]):
        r = (x + 1 for x in range(10))

        format = DillFormat[Iterable[int]](compress)
        format.write(r, self.TEST_DIR)
        r2 = format.read(self.TEST_DIR)
        assert [x + 1 for x in range(10)] == list(r2)

    @pytest.mark.parametrize("compress", _OPEN_FUNCTIONS.keys())
    def test_json_format(self, compress: Optional[str]):
        artifact = {"Hello, World!": "Hi!"}
        format = JsonFormat[Dict[str, str]](compress)
        format.write(artifact, self.TEST_DIR)
        assert format.read(self.TEST_DIR) == artifact

    @pytest.mark.parametrize("compress", _OPEN_FUNCTIONS.keys())
    def test_iterable_json_format(self, compress: Optional[str]):
        r = (x + 1 for x in range(10))

        format = JsonFormat[Iterable[int]](compress)
        format.write(r, self.TEST_DIR)
        r2 = format.read(self.TEST_DIR)
        assert [x + 1 for x in range(10)] == list(r2)
