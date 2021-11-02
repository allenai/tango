from typing import Any, Dict, Sequence

from tango import JsonFormat, Step
from tango.common import DatasetDict
from tango.common.testing import run_experiment


@Step.register("train_data")
class TrainData(Step):
    DETERMINISTIC = True
    CACHEABLE = False

    def run(self) -> Sequence[int]:
        return list(range(10))


@Step.register("val_data")
class ValData(Step):
    DETERMINISTIC = True
    CACHEABLE = False

    def run(self) -> Sequence[int]:
        return list(range(10, 20))


@Step.register("save_data")
class SaveData(Step):
    DETERMINISTIC = True
    CACHEABLE = True
    FORMAT = JsonFormat()

    def run(self, dataset_dict: DatasetDict) -> Dict[str, Any]:
        return dataset_dict.splits


def test_experiment():
    with run_experiment(
        {
            "steps": {
                "train_data": {
                    "type": "train_data",
                },
                "val_data": {
                    "type": "val_data",
                },
                "saved_data": {
                    "type": "save_data",
                    "dataset_dict": {
                        "splits": {
                            "train": {"type": "ref", "ref": "train_data"},
                            "val": {"type": "ref", "ref": "val_data"},
                        }
                    },
                },
            }
        }
    ) as run_dir:
        assert (run_dir / "saved_data").is_dir()
