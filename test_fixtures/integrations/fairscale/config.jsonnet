local pretrained_model = "sshleifer/tiny-gpt2";

####################
# Trainer settings #
####################

local training_steps = 4;
local validate_every = 4;

local devices = 2;
local grad_accum = 1;
local batch_size = 2;

local activation_checkpointing = true;
local amp = false;
local fsdp = true;
local cpu_offloading = false;  # Can only be used with 'fsdp' - saves a lot of GPU memory by offloading params+gradients to CPU, but is very slow.

######################
# Optimizer settings #
######################

local warmup_steps = 2;
local learning_rate = 0.005;


local fsdp_config = {
    reshard_after_forward: true,
    move_params_to_cpu: cpu_offloading,
    move_grads_to_cpu: cpu_offloading,
    mixed_precision: amp,
};

local training_engine = {
    type: "fairscale",
    optimizer: {
        type: "torch::AdamW",
        lr: learning_rate,
        betas: [0.9, 0.95],
        eps: 1e-6,
    },
    amp: amp,
    fsdp_config: fsdp_config,
};

local dataloader = {
  batch_size: batch_size,
  sampler: {
    type: "torch::DistributedSampler",
    shuffle: true,
    drop_last: true,
  },
};

{
    steps: {
        regression_data: {
            type: "simple_regression_data",
        },
        trained_model: {
            type: "torch::train",
            model: {
                type: "fairscale::with_wrapped_modules",
                model: {
                    type: "simple_regression_model",
                },
                modules_to_wrap: ["blocks\\.[0-9]+"],
                fsdp_config: fsdp_config,
                activation_checkpointing: activation_checkpointing,
            },
            training_engine: training_engine,
            dataset_dict: { type: "ref", ref: "regression_data" },
            train_dataloader: dataloader,
            validation_split: "dev",
            grad_accum: grad_accum,
            train_steps: training_steps,
            validate_every: training_steps,
            validation_steps: 2,
            checkpoint_every: training_steps,
            log_every: 1,
            device_count: devices,
        },
    }
}
