import os

from tango import Format
from tango.common.testing import TangoTestCase
from tango.integrations.flax.format import FlaxFormat


class TestTorchFormat(TangoTestCase):
    def test_read_write(self):
        flax_format: FlaxFormat = Format.by_name("flax")()  # type: ignore[assignment]
        flax_format.write({"a": 1}, self.TEST_DIR)
        assert os.path.exists(self.TEST_DIR / "checkpoint_0")
        data = flax_format.read(self.TEST_DIR)
        assert data == {"a": 1}
