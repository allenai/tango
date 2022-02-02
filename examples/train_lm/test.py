from tango.common import Params
from tango.common.testing import run_experiment


def test_small_experiment():
    model = "sshleifer/tiny-gpt2"
    dataloader = {
        "batch_size": 2,
        "collate_fn": {"type": "transformers::DefaultDataCollator"},
    }
    steps = 4
    overrides = {
        "steps.tokenized_data.block_size": 64,
        # Override the model in the config with the tiny alternative so training is fast.
        "steps.tokenized_data.tokenizer.pretrained_model_name_or_path": model,
        "steps.trained_model.model.model.pretrained_model_name_or_path": model,
        # Use a small number of training/validation/eval steps.
        "steps.trained_model.lr_scheduler.num_warmup_steps": 1,
        "steps.trained_model.lr_scheduler.num_training_steps": steps,
        "steps.trained_model.train_steps": steps,
        "steps.trained_model.validation_steps": 2,
        "steps.trained_model.validate_every": steps,
        "steps.final_metrics.eval_steps": 2,
        "steps.trained_model.checkpoint_every": steps,
        "steps.trained_model.device_count": 1,
        # Override data loaders.
        "steps.trained_model.train_dataloader": dataloader,
        "steps.trained_model.validation_dataloader": dataloader,
        "steps.final_metrics.dataloader": dataloader,
    }

    # Load the config.
    config = Params.from_file("config.jsonnet", params_overrides=overrides)

    # Make sure we've overrode the model entirely.
    flattened = config.as_flat_dict()
    for key, value in flattened.items():
        if "model_name" in key or (isinstance(value, str) and "gpt" in value):
            assert value == model

    with run_experiment(config, include_package=["tokenize_step.py"]) as run_dir:
        assert (run_dir / "trained_model").is_dir()
