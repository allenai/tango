{
    "steps": {
        "data_full": {
            "type": "datasets::load",
            "path": "iohadrubin/mini_xsum",
        },
        "data": {
            "type": "datasets::dataset_remix",
            "input": {"type": "ref", "ref": "data_full"},
            "new_splits": {"train": "train[:20]", "validation": "validation[:20]"},
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
                "pretrained_model_name_or_path" : "t5-small"
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
            "validate_every" : 1,
            "validation_dataloader": {
                "batch_size": 16,
                "drop_last": true
            },
            "train_epoch": 1,
            "checkpoint_every": 1,
            "log_every": 1
        }
    }
}