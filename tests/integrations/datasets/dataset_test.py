import datasets

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
        hf_dataset_dict = step.result()
        assert "train" in hf_dataset_dict
        dataset_dict = convert_to_tango_dataset_dict(hf_dataset_dict)
        assert isinstance(dataset_dict.det_hash_object(), str)

    def test_convert_to_tango_iterable_dataset_dict(self):
        hf_dataset_dict = datasets.IterableDatasetDict(
            train=datasets.iterable_dataset.iterable_dataset(({"x": x} for x in range(100)))
        )
        dataset_dict1 = convert_to_tango_dataset_dict(hf_dataset_dict)
        assert isinstance(dataset_dict1.det_hash_object(), str)
        # Doing again should produce result in a different fingerprint, and so different hash,
        # because we can never gaurantee that an iterable dataset is the same.
        dataset_dict2 = convert_to_tango_dataset_dict(hf_dataset_dict)
        assert dataset_dict1.det_hash_object() != dataset_dict2.det_hash_object()

    def test_load_concatenate_and_interleave(self):
        self.run(
            self.FIXTURES_ROOT / "integrations" / "datasets" / "config.json",
            overrides={
                "steps.train_data.cache_dir": str(self.TEST_DIR / "cache"),
                "steps.dev_data.cache_dir": str(self.TEST_DIR / "cache"),
            },
        )
