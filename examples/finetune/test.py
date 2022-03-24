# from .snli_steps import SnliText2Text
import datasets as ds

from tango.common import Params
from tango.common.testing import TangoTestCase, run_experiment


class TestSnliText2Text(TangoTestCase):
    def test_config(self):
        config = Params.from_file("test_config.jsonnet")
        with run_experiment(config, include_package=["snli_steps.py"]) as run_dir:
            assert (run_dir / "processed_data").is_dir()
            processed = ds.load_from_disk(run_dir / "processed_data" / "data")
            assert len(processed["train"][0].keys()) == 2
            assert "source" in processed["train"][0].keys()
            assert "target" in processed["train"][0].keys()
            assert processed["train"][0]["source"].startswith("nli premise:")

            assert (run_dir / "tokenized_data").is_dir()
            tokenized = ds.load_from_disk(run_dir / "tokenized_data" / "data")
            assert "input_ids" in tokenized["train"][0]

            assert (run_dir / "trained_model").is_dir()

    # def test_config_with_overrides(self):
    #     overrides = {
    #         "steps.processed_data.seq2seq": False,
    #     }
    #     config = Params.from_file("test_config.jsonnet", params_overrides=overrides)
    #     with run_experiment(config, include_package=["snli_steps.py"]) as run_dir:
    #         assert (run_dir / "processed_data").is_dir()
    #         processed = ds.load_from_disk(run_dir / "processed_data" / "data")
    #         assert len(processed["train"][0].keys()) == 1
    #         assert "source" in processed["train"][0].keys()
    #         # assert "target" in processed["train"][0].keys()
    #         assert processed["train"][0]["source"].startswith("nli premise:")


if __name__ == "__main__":
    config = Params.from_file("config.jsonnet")
    with run_experiment(
        config, include_package=["snli_steps.py", "tango.integrations.transformers.finetune"]
    ) as run_dir:
        assert (run_dir / "processed_data").is_dir()
