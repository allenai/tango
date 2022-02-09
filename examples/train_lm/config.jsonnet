##################
# Model settings #
##################

local pretrained_model = "gpt2";
# With 'fsdp' and 'activation_checkpointing' (see constants below), you should be able to train
# a 6B model on 4x ~40GB GPUs:
# local pretrained_model = "EleutherAI/gpt-j-6B";

# This doesn't seem to work with gpt2, but works fine with gpt-j.
local load_with_low_cpu_mem_usage = std.startsWith(pretrained_model, "EleutherAI/gpt-j");

####################
# Trainer settings #
####################

# Trainer settings, adjust to your use-case.
local training_steps = 200;  # total number of optimization steps to train for
local validate_every = 20;  # how often to validate and save checkpoints

local devices = 1;  # number of devices to train on (will use GPUs if enough are available, otherwise CPU)
local grad_accum = 1;  # number of gradient accumulation steps (changes the effective batch size)
# This is the batch size per GPU, ignoring gradient accumulation:
local batch_size = 8;
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


# <----- you probably don't need to edit below this line ----> #


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
    collate_fn: { type: "transformers::DefaultDataCollator" },
    sampler: {
        type: "torch::DistributedSampler",
        shuffle: true,
        drop_last: true,
    },
};

local single_device_dataloader = {
    shuffle: true,
    batch_size: batch_size,
    collate_fn: { type: "transformers::DefaultDataCollator" },
};

local dataloader = if devices > 1 then distributed_dataloader else single_device_dataloader;

{
    steps: {
        raw_data: {
            type: "datasets::load",
            path: "wikitext",
            name: "wikitext-2-raw-v1",
        },
        tokenized_data: {
            type: "tokenize_data",
            dataset: { type: "ref", ref: "raw_data" },
            tokenizer: { pretrained_model_name_or_path: pretrained_model }
        },
        trained_model: {
            type: "torch::train",
            model: {
                type: "fairscale::with_wrapped_modules",
                model: {
                    type: "transformers::AutoModelForCausalLM::from_pretrained",
                    pretrained_model_name_or_path: pretrained_model,
                    low_cpu_mem_usage: load_with_low_cpu_mem_usage,
                },
                modules_to_wrap: ["transformer\\.h\\.[0-9]+"],  # tell FairScale to wrap the transformer's blocks individually
                fsdp_config: fsdp_config,
                activation_checkpointing: activation_checkpointing,
            },
            dataset_dict: { type: "ref", ref: "tokenized_data" },
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
        final_metrics: {
            type: "torch::eval",
            model: { type: "ref", ref: "trained_model" },
            dataset_dict: { type: "ref", ref: "tokenized_data" },
            dataloader: single_device_dataloader,
            test_split: "test",
        },
    }
}
