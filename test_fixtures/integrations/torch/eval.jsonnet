{
    "steps": {
        "data": {
            "type": "generate_data",
        },
        "train": {
            "type": "torch::eval",
            "model": {
                "type": "basic_regression",
            },
            "dataset_dict": {
                "type": "ref",
                "ref": "data"
            },
            "dataloader": {
                "batch_size": 8,
                "shuffle": true
            },
            "test_split": "validation",
            "log_every": 1
        }
    }
}
