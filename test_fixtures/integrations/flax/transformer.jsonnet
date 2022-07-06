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
                "type" : "optax::sgd",
                "learning_rate" : 1e-4
            },
            "train_dataloader": {
                "batch_size": 16,
                "drop_last": true
            },
            "train_wrapper": {
                "type": "xsum_train_wrapper"
            },
            "train_split": "train",
            "validation_split" : "validation",
            "validation_dataloader": {
                "batch_size": 16,
                "drop_last": true
            },
            "train_epoch": 1,
            "checkpoint_every": 50,
            "log_every": 100
        },
        "eval": {
            "type": "flax::eval",
            "model": {
                "type" : "transformers::FlaxAutoModelForSeq2SeqLM::from_pretrained",
                "pretrained_model_name_or_path" : "facebook/bart-base"
            },
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
            "eval_wrapper": {
                "type" : "xsum_eval_wrapper"
            }
        }
    }
}