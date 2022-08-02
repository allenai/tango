from typing import Dict

from transformers import AutoTokenizer

from tango.common.testing import TangoTestCase
from tango.integrations.flax import DataLoader, FlaxDataLoader
from tango.integrations.flax.util import get_PRNGkey
from tango.step import Step


class TestDataStep(TangoTestCase):
    def test_dataloader(self) -> None:
        assert "flax::dataloader" in DataLoader.list_available()

    def test_sample_data(self) -> None:
        step = Step.from_params(  # type: ignore[assignment]
            {
                "type": "datasets::load",
                "path": "lhoestq/demo1",
                "split": "train",
                "cache_dir": str(self.TEST_DIR / "cache"),
            }
        )

        dataset = step.result()
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        column_names = dataset.column_names
        dataset = dataset.map(
            lambda e: tokenizer(e["review"], truncation=True, padding="max_length")
        )
        dataset = dataset.remove_columns(column_names)
        data = FlaxDataLoader(dataset, batch_size=16)
        rng = get_PRNGkey()
        for batch in data(rng, do_distributed=False):
            assert isinstance(batch, Dict)
