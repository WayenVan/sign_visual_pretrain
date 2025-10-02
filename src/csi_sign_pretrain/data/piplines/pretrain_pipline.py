from albumentations import (
    CenterCrop,
    Compose,
    HorizontalFlip,
    Normalize,
    RandomCrop,
    Resize,
    ColorJitter,
    ToTensorV2,
)

MEAN = (0.5, 0.5, 0.5)  # to fit the perception encoder
STD = (0.5, 0.5, 0.5)


class PretrainPipline:
    def __init__(self, height=224, width=224):
        # video transforms
        self.resize_cop = Compose(
            [
                Resize(height=256, width=256),
                RandomCrop(height=height, width=width, p=1.0),
            ],
            p=1.0,
        )

        self.warp = Compose(
            [
                ColorJitter(p=0.75),
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

        anchor = self.resize_cop(image=anchor)["image"]
        positive = self.resize_cop(image=positive)["image"]

        anchor = self.warp(image=anchor)["image"]
        positive = self.warp(image=positive)["image"]

        anchor = self.to_tensor(image=anchor)["image"]
        positive = self.to_tensor(image=positive)["image"]

        data["anchor"] = anchor
        data["positive"] = positive
        data["cached_info"] = cached_info

        return data
