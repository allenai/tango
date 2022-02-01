from typing import Any, Dict, List, Optional, Tuple

import datasets
import torch
from cached_path import cached_path
from PIL import Image
from torch import nn
from torch.optim import Adam
from torchvision import models, transforms

from tango import Step
from tango.integrations.torch import DataCollator, Model, Optimizer

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

    def forward(
        self, image: torch.Tensor, label: Optional[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        output = self.model_ft(image)
        if label is None:
            return output
        loss = self.loss_fn(output, label)
        return {"loss": loss}

# Custom data collator for images, that takes in a batch of images and labels and
# reformats the data so that it is suitable for the model.
@DataCollator.register("image_collator")
class ImageCollator(DataCollator[Tuple[torch.Tensor]]):
    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "image": torch.cat([item["image"].unsqueeze(0) for item in batch], dim=0),
            "label": torch.tensor([item["labels"] for item in batch]),
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
        "test": transforms.Compose(
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

    def pil_loader(self, path: str):
        with open(path, 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')
            transform = get_data_transforms(input_size=224)["train"]
            transformed_image = transform(image)
            return transformed_image


    def image_loader(self, example_batch):
        example_batch['image'] = [
            self.pil_loader(f) for f in example_batch['file']
        ]
        return example_batch

    def run(self, dataset: datasets, test_size: int) -> Dict[str, torch.utils.data.Dataset]:
        dataset = dataset.with_transform(self.image_loader)
        train_test = dataset["train"].train_test_split(test_size=test_size)
        return train_test


@Step.register("prediction")
class Prediction(Step):
    def run(self, image_url: str, input_size: int, model: models) -> torch.Tensor:
        # download and store image
        image_path = cached_path(image_url)
        raw_image = Image.open(image_path)

        # pass image through transform
        transform = get_data_transforms(input_size=input_size)["test"]
        transformed_image = transform(raw_image)
        transformed_image = transformed_image.unsqueeze(0)

        # pass image through model and get the prediction
        prediction = model(image=transformed_image, label=None)
        return prediction
