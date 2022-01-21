import os
from collections import defaultdict
from pickletools import optimize
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torchvision
from torch.optim import Adam
from torchvision import datasets, models, transforms

from tango import Step
from tango.integrations.torch import DataCollator, Model, Optimizer

Optimizer.register("torch_adam")(Adam)

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(num_classes: int, feature_extract: bool, use_pretrained: bool) -> models:
    model_ft = models.resnet18(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    return model_ft

Model.register("resnet_ft")(initialize_model)

@DataCollator.register("image_collator")
class ImageCollator(DataCollator[Tuple[torch.Tensor]]):
    def __call__(self, batch: List[Tuple[torch.Tensor]]) -> Dict[str, Any]:
        # data = [item[0] for item in batch]
        # target = [item[1] for item in batch]
        # return [data, target]
        return {"image": torch.cat([item[0] for item in batch], dim=0)}

@Step.register("transform_data")
class TransformData(Step):
    DETERMINISTIC = True

    def run(self, data_dir: str, input_size: int, batch_size: int) -> Dict[str, torch.utils.data.Dataset]:

        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
            'val': transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
            }

        # Create training and validation datasets
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
        # formatted_datasets = defaultdict(dict)
        # for dset, values in image_datasets:
        #     formatted_datasets[dset] = defaultdict(dict)
        #     for i, val in enumerate(values):
        #         formatted_datasets[dset][str(i)] = val
        print(image_datasets.keys())
        print(image_datasets["train"][0])
        return image_datasets


