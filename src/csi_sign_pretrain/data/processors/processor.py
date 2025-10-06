from typing import Dict, List, Optional, Union

import torch
import numpy as np
from albumentations import (
    CenterCrop,
    ColorJitter,
    Compose,
    HorizontalFlip,
    Normalize,
    RandomCrop,  # noqa: F401
    RandomResizedCrop,
    Resize,
    ToTensorV2,
)
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.utils import TensorType, filter_out_non_signature_kwargs


class SignPretrainImageProcessor(BaseImageProcessor):
    _auto_class = "AutoImageProcessor"
    model_input_names = ["anchors", "positives"]
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]

    def __init__(self, height=224, width=224, **kwargs):
        super().__init__(**kwargs)
        self.height = height
        self.width = width

    @property
    def to_tensor(self):
        return ToTensorV2()

    @property
    def anchor_transform(self):
        return Compose(
            [
                Resize(height=256, width=256),
                RandomResizedCrop(
                    size=(self.height, self.width),
                    scale=(0.3, 0.6),
                    ratio=(0.5, 1.5),
                    p=1.0,
                ),
                ColorJitter(p=0.5),
                Normalize(
                    mean=self.image_mean,
                    std=self.image_std,
                    max_pixel_value=1.0,
                ),
                HorizontalFlip(p=0.5),
            ],
            p=1.0,
        )

    @property
    def positive_transform(self):
        return Compose(
            [
                Resize(height=256, width=256),
                RandomResizedCrop(
                    size=(self.height, self.width),
                    scale=(0.5, 1.0),
                    ratio=(0.5, 1.5),
                    p=1.0,
                ),
                ColorJitter(p=0.5),
                Normalize(
                    mean=self.image_mean,
                    std=self.image_std,
                    max_pixel_value=1.0,
                ),
                HorizontalFlip(p=0.5),
            ],
            p=1.0,
        )

    def __call__(self, *args, **kwargs):
        return self.preprocess(*args, **kwargs)

    @filter_out_non_signature_kwargs()
    def preprocess(
        self,
        anchors: np.ndarray,
        positives: np.ndarray,
        **kwargs,
    ):
        augmented_anchors = []
        augmented_positives = []
        for anchor, positive in zip(anchors, positives):
            if anchor.shape != positive.shape:
                raise ValueError(
                    f"Anchor and positive must have the same shape, but got {anchor.shape} and {positive.shape}"
                )
            anchor = self.anchor_transform(image=anchor)["image"]
            positive = self.positive_transform(image=positive)["image"]

            anchor = self.to_tensor(image=anchor)["image"]
            positive = self.to_tensor(image=positive)["image"]

            augmented_anchors.append(anchor)
            augmented_positives.append(positive)

        data = dict(
            anchors=torch.stack(augmented_anchors, dim=0),
            positives=torch.stack(augmented_positives, dim=0),
        )
        return BatchFeature(data=data, tensor_type=TensorType.PYTORCH)
