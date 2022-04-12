{
    "steps": {
        "data": {
            "type": "generate_data",
        },
        "train": {
            "type": "pytorch_lightning::train",
            "model": {
                "type": "basic_regression",
            },
            "trainer": {
                "type": "default",
                "max_epochs": 5,
                "log_every_n_steps": 3,
                "logger": [
                    {"type": "pytorch_lightning::TensorBoardLogger"},
                    {"type": "pytorch_lightning::CSVLogger"},
                ],
                "accelerator": "cpu",
                "profiler": {
                    "type": "pytorch_lightning::SimpleProfiler",
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
        }
    }
}
