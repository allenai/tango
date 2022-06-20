{
    "steps": {
        "data": {
            "type": "generate_data",
        },
        "train": {
            "type": "flax::train",
            "model": {
                "type" : "classification"
            },
            "dataset": {
                "type": "ref",
                "ref": "data"
            },
            "optimizer": {
                "type" : "optax::sgd",
                "learning_rate" : 1e-4
            },
            "train_dataloader": {
                "batch_size": 4,
                "drop_last": true
            },
            "train_split": "train",
            "validation_dataloader": {
                "batch_size": 8,
                "drop_last": true,
                "shuffle": false
           },
            "validation_split": "validation",
            "train_epoch": 10,
            "validate_every": 10,
            "checkpoint_every": 100,
            "log_every": 1
        }
    }
}