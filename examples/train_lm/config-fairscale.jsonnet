local pretrained_model = "EleutherAI/gpt-j-6B";
local training_steps = 200;
local warmup_steps = 20;
local batch_size = 8;
local validate_every = 20;
local devices = 4;
local grad_accum = 2;

# FairScaleTrainEngine settings:
local fsdp = true;
local amp = true;
local activation_checkpointing = true;
local cpu_offloading = false;

# FullyShardedDataParallel config:
local fsdp_config = if fsdp then {
    reshard_after_forward: true,
    move_params_to_cpu: cpu_offloading,
    move_grads_to_cpu: cpu_offloading,
    mixed_precision: amp,
} else null;

local dataloader = {
    batch_size: batch_size,
    collate_fn: {type: "transformers_default"},
    sampler: {
        type: "torch::DistributedSampler",
        shuffle: true,
        drop_last: true,
    }
};

{
    steps: {
        raw_data: {
            type: "datasets::load",
            path: "wikitext",
            name: "wikitext-2-raw-v1",
        },
        tokenized_data: {
            type: "tokenize_data",
            dataset: {type: "ref", ref: "raw_data"},
            pretrained_model_name: pretrained_model,
        },
        trained_model: {
            type: "torch::train",
            model: {
                type: "lm_pretrained",
                pretrained_model_name_or_path: pretrained_model,
                low_cpu_mem_usage: true,
                activation_checkpointing: activation_checkpointing,
                fsdp_config: fsdp_config,
            },
            dataset_dict: {type: "ref", ref: "tokenized_data"},
            train_dataloader: dataloader,
            validation_split: "validation",
            optimizer: {
                type: "transformers_adamw",
                lr: 0.00005,
                betas: [0.9, 0.95],
                eps: 1e-6,
                correct_bias: false,
            },
            lr_scheduler: {
                type: "linear_with_warmup",
                num_warmup_steps: warmup_steps,
                num_training_steps: training_steps,
            },
            grad_accum: grad_accum,
            train_steps: training_steps,
            validate_every: validate_every,
            checkpoint_every: validate_every,
            log_every: 1,
            device_count: devices,
            callbacks: [{type: "torch::cuda_mem_stats"}],
            train_engine: {
                type: if fsdp then "fairscale" else "torch",
                amp: amp,
                [if fsdp then "fsdp_config" else null]: fsdp_config,
            },
        },
    }
}
