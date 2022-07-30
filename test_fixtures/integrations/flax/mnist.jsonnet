{
    "steps": {
        "data": {
            "type": "load_mnist_data",
        },
        "train": {
            "type": "flax::train",
            "model": {
                "type" : "mnist"
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
                "batch_size": 16,
                "drop_last": true
            },
            "train_wrapper" : {
                "type": "mnist_wrapper"
            },
            "train_split": "train",
            "train_epoch": 2,
            "checkpoint_every": 20,
            "log_every": 1
        },
        "eval": {
            "type": "flax::eval",
            "model" : {
                "type": "mnist"
            },
            "state" : {
                "type": "ref",
                "ref" : "train"
            },
            "dataset": {
                "type": "ref",
                "ref": "data"
            },
            "dataloader": {
                "batch_size": 16,
                "drop_last": true
            },
            "eval_wrapper": {
                "type": "eval_wrapper"
            }
        }
    }
}