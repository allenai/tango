from tango.common.dataset_dict import DatasetDict
from tango.steps.dataset_remix import DatasetRemixStep


def test_dataset_remix_step():
    step = DatasetRemixStep("remix")
    dataset_dict = DatasetDict(
        {
            "train": list(range(10)),
            "dev": list(range(10, 15)),
            "test": list(range(15, 20)),
        }
    )
    result = step.run(
        input=dataset_dict,
        new_splits={
            "all_train": "train + dev",
            "cross_val_train": "train[:8]",
            "cross_val_dev": "train[-2:]",
        },
    )
    assert len(result["all_train"]) == len(dataset_dict["train"]) + len(dataset_dict["dev"])
