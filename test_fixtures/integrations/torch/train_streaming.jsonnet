{
    "steps": {
        "data": {
            "type": "generate_streaming_data",
        },
        "train": {
            "type": "torch::train",
            "model": {
                "type": "basic_regression",
            },
            "dataset_dict": {
                "type": "ref",
                "ref": "data"
            },
            "train_dataloader": {
                "batch_size": 8,
                "shuffle": true
            },
            "optimizer": {
                "type": "torch::Adam",
            },
            "train_steps": 100,
            "checkpoint_every": 10,
            "log_every": 1
        }
    }
}
