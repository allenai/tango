from tango.common.testing import TangoTestCase
from tango.integrations.datasets import LoadDataset
from tango.step import Step
from tango.step_cache import DirectoryStepCache


class TestLoadDataset(TangoTestCase):
    def test_from_params(self):
        step: LoadDataset = Step.from_params(  # type: ignore[assignment]
            {
                "type": "datasets::load",
                "path": "lhoestq/test",
                "cache_dir": str(self.TEST_DIR / "cache"),
            }
        )
        dataset = step.result(DirectoryStepCache(self.TEST_DIR / "step_cache"))
        assert "train" in dataset
