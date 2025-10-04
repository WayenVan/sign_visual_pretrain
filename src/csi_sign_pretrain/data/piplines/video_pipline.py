from albumentations import (
    CenterCrop,
    Compose,
    HorizontalFlip,
    Normalize,
    RandomCrop,
    Resize,
    ColorJitter,
)
import torch

MEAN = (0.5, 0.5, 0.5)  # to fit the perception encoder
STD = (0.5, 0.5, 0.5)


class ToTensorVideo:
    def __init__(self) -> None:
        pass

    def __call__(self, video):
        video = torch.tensor(video, dtype=torch.float32)
        video = video.permute(
            0, 3, 1, 2
        )  # [time, height, width, channel] -> [time, channel, height, width]
        video = video.contiguous()
        return video


class VideoPipline:
    def __init__(self, height=224, width=224, downsample_rate=1):
        # video transforms
        self.resize_cop = Compose(
            [
                Resize(height=256, width=256),
                CenterCrop(height=height, width=width, p=1.0),
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
            ],
            p=1.0,
        )
        self.to_tensor = ToTensorVideo()

        # text transforms
        # self.delete = RandomWordAug(action="delete", aug_p=0.5)
        # self.insert = RandomWordAug(action="insert", aug_p=0.5)
        self.downsample_rate = downsample_rate

    def __call__(self, data):
        video = data["video"]
        text = data["text"]

        video = self.resize_cop(images=video)["images"]
        video = self.warp(images=video)["images"]
        video = video[:: self.downsample_rate]  # downsample video
        video = ToTensorVideo()(video)

        # text = self.delete.augment(text)[0]
        # text = self.insert.augment(text)

        data["augmented_video"] = video
        data["augmented_text"] = text

        return data
