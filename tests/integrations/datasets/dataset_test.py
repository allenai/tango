from tango.common.testing import TangoTestCase
from tango.integrations.datasets import LoadDataset
from tango.step import Step


class TestLoadDataset(TangoTestCase):
    def test_from_params(self):
        step: LoadDataset = Step.from_params(  # type: ignore[assignment]
            {
                "type": "datasets::load",
                "path": "lhoestq/test",
                "cache_dir": str(self.TEST_DIR / "cache"),
            }
        )
        dataset = step.run_with_work_dir(self.TEST_DIR / "work")
        assert "train" in dataset
