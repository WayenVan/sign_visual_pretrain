from albumentations import (
    CenterCrop,
    Compose,
    HorizontalFlip,
    Normalize,
    RandomCrop,
    Resize,
    ColorJitter,
    ToTensorV2,
    RandomResizedCrop,
)

MEAN = (0.5, 0.5, 0.5)  # to fit the perception encoder
STD = (0.5, 0.5, 0.5)


class PretrainPipline:
    def __init__(self, height=224, width=224):
        # video transforms
        self.anchor_transform = Compose(
            [
                Resize(height=256, width=256),
                RandomResizedCrop(
                    size=(height, width), scale=(0.3, 0.6), ratio=(0.5, 1.5), p=1.0
                ),
                ColorJitter(p=0.5),
                Normalize(
                    mean=MEAN,
                    std=STD,
                    max_pixel_value=1.0,
                ),
                HorizontalFlip(p=0.5),
            ],
            p=1.0,
        )
        self.positive_transform = Compose(
            [
                Resize(height=256, width=256),
                RandomResizedCrop(
                    size=(height, width), scale=(0.5, 1.0), ratio=(0.5, 1.5), p=1.0
                ),
                ColorJitter(p=0.5),
                Normalize(
                    mean=MEAN,
                    std=STD,
                    max_pixel_value=1.0,
                ),
                HorizontalFlip(p=0.5),
            ],
            p=1.0,
        )

        self.to_tensor = ToTensorV2()

    def __call__(self, data):
        anchor = data.pop("anchor", None)
        positive = data.pop("positive", None)
        cached_info = {
            "original_enchor": anchor,
            "original_positive": positive,
        }

        anchor = self.anchor_transform(image=anchor)["image"]
        positive = self.positive_transform(image=positive)["image"]

        anchor = self.to_tensor(image=anchor)["image"]
        positive = self.to_tensor(image=positive)["image"]

        data["anchor"] = anchor
        data["positive"] = positive
        data["cached_info"] = cached_info

        return data
