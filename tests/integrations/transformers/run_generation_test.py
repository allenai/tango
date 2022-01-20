from tango import Step
from tango.common import DatasetDict
from tango.common.testing import TangoTestCase
from tango.integrations.transformers import RunGenerationDataset


class TestRunGeneration(TangoTestCase):
    def test_run_generation(self):
        step = Step.from_params(  # type: ignore[assignment]
            {
                "type": "transformers::run_generation",
                "prompts": ["Tango is the future of", "Everybody should be using Tango to"],
                "model_name": "sshleifer/tiny-gpt2",
            },
        )
        result = list(step.result())
        assert len(result) == 2

    def test_run_generation_dataset(self):
        dataset = DatasetDict(
            {
                "train": [
                    {"prompt": "Tango is the future of"},
                    {"prompt": "Everybody should be using Tango to"},
                ]
            },
            {},
        )

        step = RunGenerationDataset(
            model_name="sshleifer/tiny-gpt2", input=dataset, prompt_field="prompt"
        )

        result = step.result()
        assert len(result) == 1
        train_split = result["train"]
        assert len(train_split) == 2
        assert len(train_split[1]) == 2
        assert train_split[1]["prompt"] == "Everybody should be using Tango to"
        assert all(
            g.startswith("Everybody should be using Tango to")
            for g in train_split[1]["prompt_generated"]
        )
