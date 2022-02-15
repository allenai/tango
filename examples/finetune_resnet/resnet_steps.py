from typing import Any, Dict, List, Optional

import datasets
import torch
from cached_path import cached_path
from PIL import Image
from torch import nn
from torch.optim import Adam
from torchvision import models, transforms

from tango import Format, JsonFormat, Step
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

    def forward(  # type: ignore
        self, image: torch.Tensor, label: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        output = self.model_ft(image)
        preds = torch.argmax(output, dim=1)
        if label is None:
            return {"preds": preds}
        loss = self.loss_fn(output, label)
        accuracy = (preds == label).float().mean()
        return {"loss": loss, "accuracy": accuracy}


# Custom data collator for images, that takes in a batch of images and labels and
# reformats the data so that it is suitable for the model.
@DataCollator.register("image_collator")
class ImageCollator(DataCollator[Dict[str, Any]]):
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
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


# loads and image and applies the appropriate transformation
def pil_loader(path: str, input_size: int, transform_type: str):
    with open(path, "rb") as f:
        image = Image.open(f)
        image = image.convert("RGB")
        transform = get_data_transforms(input_size=input_size)[transform_type]
        transformed_image = transform(image)
        return transformed_image


# calls the image loader on every image in a given batch
def image_loader(example_batch, input_size: int, transform_type: str):
    example_batch["image"] = [
        pil_loader(f, input_size, transform_type) for f in example_batch["file"]
    ]
    return example_batch


# This step takes in raw image data and transforms and tokenizes it.
@Step.register("transform_data")
class TransformData(Step):
    DETERMINISTIC = True
    CACHEABLE = False

    def run(  # type: ignore
        self, dataset: datasets.DatasetDict, val_size: float, input_size: int
    ) -> datasets.DatasetDict:
        def image_loader_wrapper(example_batch):
            return image_loader(example_batch, input_size=input_size, transform_type="train")

        dataset = dataset.with_transform(image_loader_wrapper)
        train_val = dataset["train"].train_test_split(test_size=val_size)
        train_val["val"] = train_val.pop("test")
        return train_val


# function to map integer labels to string labels
def convert_to_label(int_label: int) -> str:
    if int_label == 0:
        return "cat"
    else:
        return "dog"


@Step.register("prediction")
class Prediction(Step):
    FORMAT: Format = JsonFormat()

    def run(  # type: ignore
        self, image_url: str, input_size: int, model: models, device: Optional[str] = "cpu"
    ) -> Dict[str, Any]:
        # download and store image
        image_path = cached_path(image_url)
        transformed_image = pil_loader(image_path, input_size, transform_type="val")

        # pass image through transform
        transformed_image = transformed_image.unsqueeze(0).to(device)

        # pass image through model and get the prediction
        prediction = model(image=transformed_image, label=None)["preds"][0].float()
        label = convert_to_label(prediction)
        return {"image_url": image_url, "local_path": image_path, "label": label}
