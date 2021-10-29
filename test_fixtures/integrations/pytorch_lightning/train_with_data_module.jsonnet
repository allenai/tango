{
    "steps": {
        "train": {
            "type": "pytorch_lightning::train",
            "model": {
                "type": "basic_regression",
            },
            "datamodule": {
                "type": "generate_data_module",
                "batch_size": 8,
                "eval_batch_size": 8,
                "shuffle": true,
                "eval_shuffle": false,
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
        }
    }
}
