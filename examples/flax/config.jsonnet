{
    "steps": {
        "data": {
            "type": "datasets::load",
            "path": "xsum",
        },
        "tokenize": {
            "type": "tokenize_data",
            "dataset": {
                "type": "ref",
                "ref": "data"
            }
        },
        "train": {
            "type": "flax::train",
            "model": {
                "type" : "transformers::FlaxAutoModelForSeq2SeqLM::from_pretrained",
                "pretrained_model_name_or_path" : "facebook/bart-base"
            },
            "dataset": {
                "type": "ref",
                "ref": "tokenize"
            },
            "optimizer": {
                "type" : "optax::adamw",
                "learning_rate" : 2e-5
            },
            "train_dataloader": {
                "batch_size": 16,
                "drop_last": true
            },
            "wrapper": {
                "type": "xsum_wrapper"
            },
            "train_split": "train",
            "validation_split" : "validation",
            "validate_every" : 1000,
            "validation_dataloader": {
                "batch_size": 16,
                "drop_last": true
            },
            "train_epoch": 5,
            "checkpoint_every": 1000,
            "log_every": 1000,

            "callbacks" : [{
                "type" : "wandb::log_flax"
            },
            {
                "type": "flax::generate_step"
            }]
        },
        "eval": {
            "type": "flax::eval",
            "state": {
                "type": "ref",
                "ref": "train"
            },
            "dataset": {
                "type": "ref",
                "ref" : "tokenize"
            },
            "dataloader": {
                "batch_size": 16,
                "drop_last": true
            },
            "wrapper": {
                "type" : "xsum_wrapper"
            }
        }
    }
}