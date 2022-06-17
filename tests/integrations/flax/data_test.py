from typing import Dict

import jaxlib.xla_extension
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from tango.integrations.flax import DataLoader, FlaxDataLoader
from tango.integrations.flax.util import get_PRNGkey


def test_dataloader() -> None:
    assert "flax::dataloader" in DataLoader.list_available()


def test_load_numpy() -> None:
    dataset = np.random.rand(100, 5)
    rng = get_PRNGkey()
    data = FlaxDataLoader(dataset, batch_size=16)

    for batch in data(rng):
        assert isinstance(batch, jaxlib.xla_extension.DeviceArray)


def test_load_dataset() -> None:
    dataset = load_dataset("glue", "mrpc", split="train")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    dataset = dataset.map(
        lambda e: tokenizer(e["sentence1"], truncation=True, padding="max_length")
    )
    dataset = dataset.remove_columns(["sentence1", "sentence2", "idx"])
    data = FlaxDataLoader(dataset, batch_size=16)
    rng = get_PRNGkey()

    for batch in data(rng):
        assert isinstance(batch, Dict)
        for val in batch.values():
            assert isinstance(val, jaxlib.xla_extension.DeviceArray)
