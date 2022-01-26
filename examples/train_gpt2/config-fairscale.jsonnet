local pretrained_model = "gpt2";
local training_steps = 200;
local warmup_steps = 20;
local batch_size = 8;
local validate_every = 20;
local distributed = true;  # Set to `true` to train on 2 (or more) GPUs.
local devices = if distributed then 4 else 1;
local grad_accum = if distributed then 2;

local distributed_dataloader = {
    "batch_size": batch_size,
    "collate_fn": {"type": "transformers_default"},
    "sampler": {
        "type": "torch::DistributedSampler",
        "shuffle": true,
        "drop_last": true,
    }
};

local single_device_dataloader = {
    "shuffle": true,
    "batch_size": batch_size,
    "collate_fn": {"type": "transformers_default"},
};

local dataloader = if distributed then distributed_dataloader else single_device_dataloader;

{
    "steps": {
        "raw_data": {
            "type": "datasets::load",
            "path": "wikitext",
            "name": "wikitext-2-raw-v1",
        },
        "tokenized_data": {
            "type": "tokenize_data",
            "dataset": {"type": "ref", "ref": "raw_data"},
            "pretrained_model_name": pretrained_model,
        },
        "trained_model": {
            "type": "torch::train",
            "model": {
                "type": "gpt2-random",
                "pretrained_model_name_or_path": pretrained_model,
                "fsdp": true,
            },
            "dataset_dict": {"type": "ref", "ref": "tokenized_data"},
            "train_dataloader": dataloader,
            "validation_split": "validation",
            "optimizer": {
                "type": "transformers_adamw",
                "lr": 0.0007,
                "betas": [0.9, 0.95],
                "eps": 1e-6,
                "correct_bias": false,
            },
            "lr_scheduler": {
                "type": "linear_with_warmup",
                "num_warmup_steps": warmup_steps,
                "num_training_steps": training_steps,
            },
            "grad_accum": grad_accum,
            "train_steps": training_steps,
            "validate_every": validate_every,
            "checkpoint_every": validate_every,
            "log_every": 1,
            "device_count": devices,
            "callbacks": [{"type": "torch::cuda_mem_stats"}],
            "accelerator": {
                "type": "fairscale",
            },
        },
        "final_metrics": {
            "type": "torch::eval",
            "model": {"type": "ref", "ref": "trained_model"},
            "dataset_dict": {"type": "ref", "ref": "tokenized_data"},
            "dataloader": single_device_dataloader,
            "test_split": "test",
        },
    }
}
