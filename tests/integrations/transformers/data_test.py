from transformers.data.data_collator import DataCollatorWithPadding, DefaultDataCollator

from tango.integrations.torch import DataCollator
from tango.integrations.transformers.data import *  # noqa: F403,F401


def test_init_collator_no_tokenizer():
    collator = DataCollator.from_params({"type": "transformers::DefaultDataCollator"})
    assert isinstance(collator, DefaultDataCollator)


def test_init_collator_with_tokenizer():
    collator = DataCollator.from_params(
        {
            "type": "transformers::DataCollatorWithPadding",
            "tokenizer": {
                "pretrained_model_name_or_path": "epwalsh/bert-xsmall-dummy",
            },
        }
    )
    assert isinstance(collator, DataCollatorWithPadding)
