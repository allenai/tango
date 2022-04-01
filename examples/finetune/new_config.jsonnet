##################
# Model settings #
##################

local pretrained_model = "patrickvonplaten/t5-tiny-random";
local load_with_low_cpu_mem_usage = false;

local modules_to_wrap = ["encoder\\.block\\.[0-9]+", "decoder\\.block\\.[0-9]+"];  # tell FairScale to wrap the transformer's blocks individually

####################
# Trainer settings #
####################

# Trainer settings, adjust to your use-case.
local training_steps = 20;  # total number of optimization steps to train for
local validate_every = 5;  # how often to validate and save checkpoints

local devices = 1;  # number of devices to train on (will use GPUs if enough are available, otherwise CPU)
local grad_accum = 1;  # number of gradient accumulation steps (changes the effective batch size)
# This is the batch size per GPU, ignoring gradient accumulation:
local batch_size = 2;
# So the effective batch size is `batch_size * grad_accum * devices`

local activation_checkpointing = false;  # use activation/gradient checkpointing (probably need this GPT-J 6B, but not gpt2)
local amp = false;  # use PyTorch's native automatic mixed precision
local fsdp = false;  # Use FairScale's FullyShardedDataParallel (probably need this GPT-J 6B, but not gpt2)
local cpu_offloading = false;  # Can only be used with 'fsdp' - saves a lot of GPU memory by offloading params+gradients to CPU, but is very slow.

######################
# Optimizer settings #
######################

local warmup_steps = 20;
local learning_rate = 0.00005;  # you can probably use a higher LR for a small model like "gpt2"


assert fsdp == true || cpu_offloading == false : "cpu_offloading only available with fsdp";

# FullyShardedDataParallel config:
local fsdp_config = if fsdp then {
    reshard_after_forward: true,
    move_params_to_cpu: cpu_offloading,
    move_grads_to_cpu: cpu_offloading,
    mixed_precision: amp,
} else null;

local training_engine = {
    type: if fsdp then "fairscale" else "torch",
    optimizer: {
        type: "torch::AdamW",
        lr: learning_rate,
        betas: [0.9, 0.95],
        eps: 1e-6,
    },
    lr_scheduler: {
        type: "transformers::linear",
        num_warmup_steps: warmup_steps,
        num_training_steps: training_steps,
    },
    amp: amp,
    [if fsdp then "fsdp_config" else null]: fsdp_config,
};

local distributed_dataloader = {
    batch_size: batch_size,
    sampler: {
        type: "torch::DistributedSampler",
        shuffle: true,
        drop_last: true,
    },
};

local single_device_dataloader = {
    shuffle: true,
    batch_size: batch_size,
};

local dataloader = if devices > 1 then distributed_dataloader else single_device_dataloader;

{
    steps: {
        raw_data: {
            type: "datasets::load",
            path: "snli",
        },
        "subset_data": {
            type: "subset-data",
            data: { type: "ref", ref: "raw_data" },
            max_samples: 10,
        },
        processed_data: {
            type: "snli-text2text",
            data: { type: "ref", ref: "subset_data" },
        },
        trained_model: {
            type: "transformers::finetune",
            model: {
                type: "fairscale::with_wrapped_modules",
                model: {
                    type: "transformers::finetune::from_pretrained",
                    pretrained_model_name_or_path: pretrained_model,
                    low_cpu_mem_usage: load_with_low_cpu_mem_usage,
                },
                modules_to_wrap: modules_to_wrap,  # tell FairScale to wrap the transformer's blocks individually
                fsdp_config: fsdp_config,
                activation_checkpointing: activation_checkpointing,
            },
            tokenizer: {
                pretrained_model_name_or_path: pretrained_model
            },
            dataset_dict: { type: "ref", ref: "processed_data" },
            train_dataloader: dataloader,
            validation_split: "validation",
            grad_accum: grad_accum,
            train_steps: training_steps,
            validate_every: validate_every,
            checkpoint_every: validate_every,
            log_every: 1,
            device_count: devices,
            training_engine: training_engine,
        },
        "generations": {
            "type": "transformers::run_generation_dataset",
            "max_length": 5,
            "input": {"type": "ref", "ref": "processed_data"},
            "batch_size": batch_size,
            "model": {"type": "ref", "ref": "trained_model"},
            "prompt_field": "source",
            "output_field": "generation",
            "splits": ["validation"]
        },
    }
}
