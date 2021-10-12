local pretrained_model = "gpt2";
local training_steps = 60;
local batch_size = 8;
local checkpoint_every = 20;

{
    "steps": {
        "raw_data": {
            "type": "datasets::load",
            "path": "wikitext",
            "name": "wikitext-2-raw-v1",
        },
        "tokenized_data": {
            "type": "tokenize_data",
            "dataset": {"type": "ref", "ref": "raw_data"},
            "pretrained_model_name": pretrained_model,
        },
        "trained_model": {
            "type": "torch::train",
            "model": {
                "type": "gpt2",
                "pretrained_model_name_or_path": pretrained_model,
            },
            "dataset_dict": {"type": "ref", "ref": "tokenized_data"},
            "train_dataloader": {
                "shuffle": true,
                "batch_size": batch_size,
                "collate_fn": {"type": "transformers_default"},
            },
            "validation_dataloader": {
                "batch_size": batch_size,
                "collate_fn": {"type": "transformers_default"},
            },
            "validation_split": "validation",
            "optimizer": {
                "type": "transformers_adamw",
                "lr": 0.0021,
                "betas": [0.9, 0.95],
                "eps": 1e-6,
                "correct_bias": false,
            },
            "lr_scheduler": {
                "type": "linear_with_warmup",
                "num_warmup_steps": 10,
                "num_training_steps": training_steps,
            },
            "train_steps": training_steps,
            "validate_every": checkpoint_every,
            "checkpoint_every": checkpoint_every,
            "log_every": 1,
        }
    }
}
