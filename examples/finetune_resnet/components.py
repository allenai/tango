import imp
import os
from collections import defaultdict
from pickletools import optimize
from typing import Any, Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import datasets, models, transforms

from tango import Step
from tango.integrations.torch import DataCollator, Model, Optimizer
from cached_path import cached_path

from PIL import Image


# Register the Adam optimizer as an `Optimizer` so we can use it in the train step.
Optimizer.register("torch_adam")(Adam)

# Wrapper class around the pre-trained ResNet-18 model that modifies the final layer.
@Model.register("resnet_ft")
class ResNetWrapper(Model):

    def __init__(self, num_classes: int, feature_extract: bool, use_pretrained: bool):
        super().__init__()
        self.model_ft = models.resnet18(pretrained=use_pretrained)
        self.set_parameter_requires_grad(self.model_ft, feature_extract)
        num_features = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(num_features, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def set_parameter_requires_grad(self, model: models, feature_extracting: bool):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    def forward(self, image: torch.Tensor, label: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        output = self.model_ft(image)  # tensor
        if not label:
            return output
        loss = self.loss_fn(output, label)
        return {"loss": loss}

# Custom data collator for images, that takes in a batch of images and labels and
# reformats the data so that it is suitable for the model.
@DataCollator.register("image_collator")
class ImageCollator(DataCollator[Tuple[torch.Tensor]]):
    def __call__(self, batch: List[Tuple[torch.Tensor]]) -> Dict[str, Any]:
        return {
            "image": torch.cat([item[0].unsqueeze(0) for item in batch], dim=0),
            "label": torch.tensor([item[1] for item in batch]),
        }

# Function that returns an image transformations dict with the appropriate image size.
def get_data_transforms(input_size: int):
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }
    return data_transforms
    
# This step takes in raw image data and transforms and tokenizes it.
@Step.register("transform_data")
class TransformData(Step):
    DETERMINISTIC = True
    CACHEABLE = False

    def run(
        self, data_dir: str, input_size: int, batch_size: int
    ) -> Dict[str, torch.utils.data.Dataset]:

        # Create training and validation datasets
        image_datasets = {
            x: datasets.ImageFolder(
                os.path.join(data_dir, x), 
                get_data_transforms(input_size=input_size)[x])
            for x in ["train", "val"]
        }
        return image_datasets

@Step.register("prediction")
class Prediction(Step):
    def run(self, image_url: str, input_size: int, model: models) -> torch.Tensor:
        # download and store image
        image_path = cached_path(image_url)
        raw_image = Image.open(image_path)
        
        # pass image through transform
        transform = get_data_transforms(input_size=input_size)["val"]
        transformed_image = transform(raw_image)
        prediction = model(transformed_image)
        return prediction
