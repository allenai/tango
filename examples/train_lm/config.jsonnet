local pretrained_model = "gpt2";
# With 'fsdp' and 'activation_checkpointing' (see constants below), you should be able to train
# a 6B model on 4x ~40GB GPUs:
# local pretrained_model = "EleutherAI/gpt-j-6B";

# Trainer settings, adjust to your use-case.
local training_steps = 200;
local warmup_steps = 20;
local batch_size = 8;
local validate_every = 20;
local devices = 4;  # number of devices to train on
local grad_accum = 2;
local activation_checkpointing = true;
local amp = true;
local fsdp = true;  # Use FairScale's FullyShardedDataParallel
local cpu_offloading = false;  # Can only be used with 'fsdp' - saves a lot of GPU memory but is very slow.

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
                type: "lm_pretrained",
                pretrained_model_name_or_path: pretrained_model,
                low_cpu_mem_usage: true,
                activation_checkpointing: activation_checkpointing,
                fsdp_config: fsdp_config,
            },
            dataset_dict: { type: "ref", ref: "tokenized_data" },
            train_dataloader: dataloader,
            validation_split: "validation",
            optimizer: {
                type: "torch::AdamW",
                lr: 0.00005,
                betas: [0.9, 0.95],
                eps: 1e-6,
            },
            lr_scheduler: {
                type: "transformers::linear",
                num_warmup_steps: warmup_steps,
                num_training_steps: training_steps,
            },
            grad_accum: grad_accum,
            train_steps: training_steps,
            validate_every: validate_every,
            checkpoint_every: validate_every,
            log_every: 1,
            device_count: devices,
            callbacks: [{ type: "torch::cuda_mem_stats" }],
            training_engine: {
                type: if fsdp then "fairscale" else "torch",
                amp: amp,
                [if fsdp then "fsdp_config" else null]: fsdp_config,
            },
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
