from tango.common.testing import TangoTestCase
from tango.integrations.datasets import LoadDataset, convert_to_tango_dataset_dict
from tango.step import Step


class TestDatasets(TangoTestCase):
    def test_from_params_and_convert_to_tango_dataset_dict(self):
        step: LoadDataset = Step.from_params(  # type: ignore[assignment]
            {
                "type": "datasets::load",
                "path": "lhoestq/test",
                "cache_dir": str(self.TEST_DIR / "cache"),
            }
        )
        hf_dataset_dict = step.run_with_work_dir(self.TEST_DIR / "work")
        assert "train" in hf_dataset_dict
        dataset_dict = convert_to_tango_dataset_dict(hf_dataset_dict)
        assert isinstance(dataset_dict.det_hash_object(), str)

    def test_load_concatenate_and_interleave(self):
        self.run(
            self.FIXTURES_ROOT / "integrations" / "datasets" / "config.json",
            overrides={
                "steps.train_data.cache_dir": str(self.TEST_DIR / "cache"),
                "steps.dev_data.cache_dir": str(self.TEST_DIR / "cache"),
            },
        )
