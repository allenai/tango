from tango.common.dataset_dict import DatasetDict


def test_dataset_dict():
    dataset_dict = DatasetDict(splits={"train": list(range(10)), "test": list(range(5))})
    assert len(dataset_dict) == 2
    assert "train" in dataset_dict
    assert "test" in dataset_dict
    assert len(dataset_dict["train"]) == 10
    assert len(dataset_dict["test"]) == 5
    assert set(dataset_dict) == set(dataset_dict.keys()) == {"train", "test"}
