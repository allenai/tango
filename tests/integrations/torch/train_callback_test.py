from pathlib import Path

import pytest
from torch.optim import SGD

from tango.common import DatasetDict, Lazy
from tango.integrations.torch import (
    DataLoader,
    StopEarly,
    StopEarlyCallback,
    TorchTrainingEngine,
    TrainConfig,
)
from tango.workspaces import MemoryWorkspace

from .training_engine_test import DummyModel


def test_stop_early_callback():
    workspace = MemoryWorkspace()
    train_config = TrainConfig(step_id="FakeStep-abc123", work_dir=Path("/tmp"))
    training_engine = TorchTrainingEngine(
        train_config=train_config, model=DummyModel(), optimizer=Lazy(SGD, lr=0.001)
    )
    dataset_dict = DatasetDict(splits={"train": [1, 2, 3]})
    train_dataloader = Lazy(DataLoader)

    callback = StopEarlyCallback(
        patience=10,
        workspace=workspace,
        train_config=train_config,
        training_engine=training_engine,
        dataset_dict=dataset_dict,
        train_dataloader=train_dataloader,
    )
    callback.post_val_loop(1, 1, 0.5, 0.5)
    callback.post_val_loop(2, 1, 0.5, 0.5)
    callback.post_val_loop(20, 1, 0.6, 0.6)
    with pytest.raises(StopEarly):
        callback.post_val_loop(31, 1, 0.6, 0.6)
