import os

from tango import Format
from tango.common.testing import TangoTestCase
from tango.integrations.torch.format import TorchFormat


class TestTorchFormat(TangoTestCase):
    def test_read_write(self):
        torch_format: TorchFormat = Format.by_name("torch")()  # type: ignore[assignment]
        torch_format.write({"a": 1}, self.TEST_DIR)
        assert os.path.exists(self.TEST_DIR / "data.pt")
        data = torch_format.read(self.TEST_DIR)
        assert data == {"a": 1}
