from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from tango.common.testing import TangoTestCase
from tango.integrations.transformers import TokenizeText2TextData


class TestTokenizeText2TextData(TangoTestCase):
    def test_tokenize_seq2seq(self):
        dataset = Dataset.from_dict(
            {"field1": ["hello", "hi"], "field2": ["world", "me"], "meta_field": [1, 0]}
        )
        data_dict = DatasetDict({"train": dataset})
        tokenizer = AutoTokenizer.from_pretrained("patrickvonplaten/t5-tiny-random")
        step = TokenizeText2TextData()

        tokenized = step.run(
            data=data_dict, tokenizer=tokenizer, source_field="field1", target_field="field2"
        )
        assert isinstance(tokenized, DatasetDict)
        assert len(tokenized["train"]) == 2
        assert "input_ids" in tokenized["train"].column_names
        assert "labels" in tokenized["train"].column_names
        assert tokenized["train"][0]["input_ids"] == [21820, 1]

    def test_tokenize_concat(self):
        dataset = Dataset.from_dict(
            {"field1": ["hello", "hi"], "field2": ["world", "me"], "meta_field": [1, 0]}
        )
        data_dict = DatasetDict({"train": dataset})
        tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
        step = TokenizeText2TextData()

        tokenized = step.run(
            data=data_dict,
            tokenizer=tokenizer,
            source_field="field1",
            target_field="field2",
            concat_source_target=True,
        )
        assert isinstance(tokenized, DatasetDict)
        assert len(tokenized["train"]) == 2
        assert "input_ids" in tokenized["train"].column_names
        assert "labels" in tokenized["train"].column_names
        assert tokenized["train"][0]["input_ids"] == [31373, 50257, 6894, 50256]
        assert tokenized["train"][0]["labels"] == [-100, -100, 6894, 50256]
