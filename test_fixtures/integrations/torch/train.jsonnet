{
    "steps": {
        "data": {
            "type": "generate_data",
        },
        "train": {
            "type": "torch::train",
            "model": {
                "type": "basic_regression",
            },
            "training_engine": {
                "optimizer": {
                    "type": "torch::Adam",
                },
            },
            "dataset_dict": {
                "type": "ref",
                "ref": "data"
            },
            "train_dataloader": {
                "batch_size": 8,
                "shuffle": true
            },
            "validation_split": "validation",
            "validation_dataloader": {
                "batch_size": 8,
                "shuffle": false
            },
            "train_steps": 100,
            "validate_every": 10,
            "checkpoint_every": 10,
            "log_every": 1
        }
    }
}
