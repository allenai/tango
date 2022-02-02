from tango import Step
from tango.common import DatasetDict
from tango.integrations.transformers import Tokenizer

LOREM_IPSUM = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut
labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris
nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit
esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt
in culpa qui officia deserunt mollit anim id est laborum.
""".replace(
    "\n", " "
).strip()


@Step.register("fairscale_test_load_data")
class FairScaleTestLoadDataStep(Step):
    DETERMINISTIC = True
    CACHEABLE = False

    def run(self, tokenizer: Tokenizer, block_size: int = 8) -> DatasetDict:
        # Tokenize text.
        tokenized = tokenizer(LOREM_IPSUM)
        tokenized["labels"] = tokenized["input_ids"].copy()
        total_length = len(tokenized["input_ids"])
        assert total_length > (block_size * 2)
        total_length = total_length // block_size
        blocks = [
            {key: x[i : i + block_size] for key, x in tokenized.items()}
            for i in range(0, total_length, block_size)
        ]
        dataset_dict = DatasetDict(splits={"train": blocks, "dev": blocks})
        return dataset_dict
