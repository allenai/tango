# from .snli_steps import SnliText2Text
import datasets as ds

from tango.common import Params
from tango.common.testing import TangoTestCase, run_experiment


class TestSnliText2Text(TangoTestCase):
    def test_config_with_t5(self):
        model = "patrickvonplaten/t5-tiny-random"
        overrides = {
            "steps.trained_model.model.model.pretrained_model_name_or_path": model,
        }
        config = Params.from_file("config.jsonnet", params_overrides=overrides)
        # Make sure we've overrode the model entirely.
        flattened = config.as_flat_dict()
        for key, value in flattened.items():
            if "model_name" in key or (isinstance(value, str) and "t5" in value):
                assert value == model

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

    def test_config_with_gpt2(self):
        model = "sshleifer/tiny-gpt2"
        overrides = {
            "steps.trained_model.model.model.pretrained_model_name_or_path": model,
            "steps.tokenized_data.tokenizer.pretrained_model_name_or_path": model,
            "steps.trained_model.train_dataloader.collate_fn.tokenizer.pretrained_model_name_or_path": model,
            "steps.final_metrics.dataloader.collate_fn.tokenizer.pretrained_model_name_or_path": model,
        }
        config = Params.from_file("config.jsonnet", params_overrides=overrides)
        # Make sure we've overrode the model entirely.
        flattened = config.as_flat_dict()
        for key, value in flattened.items():
            if "model_name" in key or (isinstance(value, str) and "gpt2" in value):
                assert value == model, key

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
