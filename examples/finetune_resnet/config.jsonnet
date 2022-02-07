local input_size = 224;
local batch_size = 32;
local num_classes = 2;
local test_size = 0.2;
local model = "resnet";
local feature_extract = true;
local distributed = false;
local devices = if distributed then 2 else 1;
local pretrained_model = "resnet_ft";
local training_steps = 500;
local validate_every = 50;
local image_url = "https://tinyurl.com/2p9xjvn9";

local distributed_dataloader = {
    batch_size: batch_size,
    sampler: {
        type: "torch::DistributedSampler",
        shuffle: true,
        drop_last: true,
    },
    collate_fn: {"type": "image_collator"},
};

local single_device_dataloader = {
    shuffle: true,
    batch_size: batch_size,
    collate_fn: {"type": "image_collator"},
};

{
    steps: {
        raw_data: {
            type: "datasets::load",
            path: "nateraw/auto-cats-and-dogs",
            name: "cats_and_dogs",
        },
        transform_data: {
            type: "transform_data",
            dataset: { type: 'ref', ref: 'raw_data' },
            input_size: input_size,
            test_size: test_size,
        },
        trained_model: {
            type: "torch::train",
            model: {
                type: pretrained_model,
                num_classes: num_classes,
                feature_extract: true,
                use_pretrained: true,
            },
            dataset_dict: {"type": "ref", "ref": "transform_data"},
            train_dataloader: single_device_dataloader,
            validation_split: null,
            optimizer: {
                type: "torch_adam",
                lr: 0.001,
            },
            train_steps: training_steps,
            validate_every: validate_every,
            checkpoint_every: validate_every,
            log_every: 1,
            device_count: devices,
        },
        final_metrics: {
            type: "torch::eval",
            model: {"type": "ref", "ref": "trained_model"},
            dataset_dict: {"type": "ref", "ref": "transform_data"},
            dataloader: single_device_dataloader,
            test_split: "test",
        },
        prediction: {
            type: "prediction",
            image_url: image_url,
            input_size: input_size,
            model: {"type": "ref", "ref": "trained_model"},
        },
    },
}
 
