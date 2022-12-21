import datasets

from tango.common.sequences import MappedSequence
from tango.common.testing import TangoTestCase
from tango.integrations.datasets import (
    DatasetRemixStep,
    DatasetsFormat,
    LoadDataset,
    convert_to_tango_dataset_dict,
)
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
        assert "train" in dataset_dict.splits

    def test_convert_to_tango_iterable_dataset_dict(self):
        hf_dataset_dict = datasets.IterableDatasetDict(
            train=datasets.iterable_dataset.IterableDataset.from_generator(
                ({"x": x} for x in range(100))
            )
        )
        dataset_dict1 = convert_to_tango_dataset_dict(hf_dataset_dict)
        assert "train" in dataset_dict1.splits

    def test_load_concatenate_and_interleave(self):
        result_dir = self.run(
            self.FIXTURES_ROOT / "integrations" / "datasets" / "config.json",
            overrides={
                "steps.train_data.cache_dir": str(self.TEST_DIR / "cache"),
                "steps.dev_data.cache_dir": str(self.TEST_DIR / "cache"),
            },
        )
        assert (result_dir / "train_data" / "data").is_dir()
        dataset = DatasetsFormat().read(result_dir / "train_data")
        assert len(dataset) == 2


def test_mapped_sequence_of_dataset():
    ds = datasets.load_dataset("piqa", split="validation")
    mapped_ds = MappedSequence(lambda x: x["goal"], ds)  # type: ignore[arg-type]
    assert len(ds) == len(mapped_ds)  # type: ignore[arg-type]
    assert ds[0]["goal"] == mapped_ds[0]  # type: ignore[index]
    assert ds[0]["goal"] == mapped_ds[:10][0]  # type: ignore[index]


def test_datasets_dataset_remix():
    dataset_dict = datasets.load_dataset("lhoestq/test")
    step = DatasetRemixStep()
    result = step.run(
        input=dataset_dict,  # type: ignore[arg-type]
        new_splits={
            "all": "train + validation",
            "crossval_train": "train[:1] + validation[1:]",
            "crossval_test": "train[1:] + validation[:1]",
        },
    )
    assert len(result["all"]) == len(dataset_dict["train"]) + len(dataset_dict["validation"])  # type: ignore
    assert len(result["crossval_train"]) == 3
    assert len(result["crossval_test"]) == 2
