from tango.common.testing import TangoTestCase
from tango.integrations.datasets import HuggingFaceDataset
from tango.step import Step


class TestHuggingFaceDataset(TangoTestCase):
    def test_from_params(self):
        step: HuggingFaceDataset = Step.from_params(  # type: ignore[assignment]
            {
                "type": "hf_dataset",
                "path": "lhoestq/test",
                "cache_dir": str(self.TEST_DIR / "cache"),
            }
        )
        dataset = step.result()
        assert "train" in dataset
