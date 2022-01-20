import copy
import imp
import os
from pickletools import optimize
import time
from typing import Any, Dict, List

import datasets
import torch
import torch.nn as nn
import torchvision
from torch.optim import Adam
from torchvision import datasets as torch_datasets
from torchvision import models, transforms

from tango import Step
from tango.integrations.datasets import DatasetsFormat
from tango.integrations.torch import DataCollator, LRScheduler, Model, Optimizer


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

Model.register("resnet_ft")(initialize_model)


@Step.register("transform_data")
class TransformData(Step):
    DETERMINISTIC = True
    CACHEABLE = True
    FORMAT = DatasetsFormat()

    def run(self, data_dir: str, input_size: int, batch_size: int) -> datasets.DatasetDict:

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
        image_datasets = {x: torch_datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
        return image_datasets





@Step.register("print_data")
class PrintData(Step):
    def run(self, dataset: datasets.DatasetDict) -> datasets.DatasetDict:
        for key, val in dataset.items():
            print(key, len(val))


