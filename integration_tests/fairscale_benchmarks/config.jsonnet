##################
# Model settings #
##################

local pretrained_model = "EleutherAI/gpt-j-6B";
local load_with_low_cpu_mem_usage = true;  # be careful, this doesn't seem to work with every model

####################
# Trainer settings #
####################

# Trainer settings, adjust to your use-case.
local training_steps = 200;  # total number of optimization steps to train for
local validate_every = 50;  # how often to validate and save checkpoints

local devices = 4;
local grad_accum = 1;  # number of gradient accumulation steps (changes the effective batch size)
# This is the batch size per GPU, ignoring gradient accumulation:
local batch_size = 8;
# So the effective batch size is `batch_size * grad_accum * devices`

######################
# Optimizer settings #
######################

local warmup_steps = 20;
local learning_rate = 0.00005;


# <----- you probably don't need to edit below this line ----> #


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

local TrainStep(options) =
    local training_engine = {
        type: if options.fsdp_config != null then "fairscale" else "torch",
        amp: options.amp,
        [if options.fsdp_config != null then "fsdp_config" else null]: options.fsdp_config,
    };

    {
        type: "torch::train",
        model: {
            type: "fairscale::with_wrapped_modules",
            model: {
                type: "transformers::AutoModelForCausalLM::from_pretrained",
                pretrained_model_name_or_path: pretrained_model,
                low_cpu_mem_usage: load_with_low_cpu_mem_usage,
            },
            modules_to_wrap: ["transformer\\.h\\.[0-9]+"],  # tell FairScale to wrap the transformer's blocks individually
            fsdp_config: options.fsdp_config,
            activation_checkpointing: options.activation_checkpointing,
        },
        dataset_dict: { type: "ref", ref: "tokenized_data" },
        train_dataloader: distributed_dataloader,
        validation_split: "validation",
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
        grad_accum: grad_accum,
        train_steps: training_steps,
        validate_every: validate_every,
        checkpoint_every: validate_every,
        log_every: 1,
        device_count: devices,
        training_engine: training_engine,
        callbacks: [
            {
                type: "torch::cuda_mem_stats",
            },
            {
                type: "wandb::log",
                entity: "allennlp",
                project: "tango-fairscale-benchmarks",
                wandb_config: options,
            },
        ],
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
            dataset: { type: "ref", ref: "raw_data" },
            tokenizer: { pretrained_model_name_or_path: pretrained_model }
        },
    } + {
        ["trained_model_" + options.name]: TrainStep(options)
        for options in [
            {
                name: "baseline",
                amp: false,
                fsdp_config: null,
                activation_checkpointing: false,
            },
            {
                name: "amp_and_checkpointing_and_fsdp_mp_with_offloading",
                amp: true,
                activation_checkpointing: true,
                fsdp_config: {
                    reshard_after_forward: true,
                    move_params_to_cpu: true,
                    move_grads_to_cpu: true,
                    mixed_precision: true,
                },
            },
        ]
    }
}
