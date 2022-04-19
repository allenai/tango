import datasets as ds
import pytest

from tango.common import Params
from tango.common.testing import TangoTestCase, run_experiment


class TestFinetuneSNLI(TangoTestCase):
    @pytest.mark.parametrize(
        "model, model_type",
        [("patrickvonplaten/t5-tiny-random", "t5"), ("sshleifer/tiny-gpt2", "gpt2")],
    )
    def test_config(self, model: str, model_type: str):
        overrides = {
            "steps.trained_model.model.model.pretrained_model_name_or_path": model,
            "steps.trained_model.tokenizer.pretrained_model_name_or_path": model,
            "steps.subset_data": {
                "type": "subset-data",
                "data": {"type": "ref", "ref": "raw_data"},
                "max_samples": 10,
            },
            "steps.processed_data.data.ref": "subset_data",
        }
        config = Params.from_file("config.jsonnet", params_overrides=overrides)
        # Make sure we've overrode the model entirely.
        flattened = config.as_flat_dict()
        for key, value in flattened.items():
            if "model_name" in key or (isinstance(value, str) and model_type in value):
                assert value == model

        with run_experiment(config, include_package=["snli_steps.py"]) as run_dir:
            assert (run_dir / "processed_data").is_dir()
            processed = ds.load_from_disk(run_dir / "processed_data" / "data")
            assert len(processed["train"][0].keys()) == 2
            assert "source" in processed["train"][0].keys()
            assert "target" in processed["train"][0].keys()
            assert processed["train"][0]["source"].startswith("nli premise:")

            assert (run_dir / "trained_model").is_dir()
