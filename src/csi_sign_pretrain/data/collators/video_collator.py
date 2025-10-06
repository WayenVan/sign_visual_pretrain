import numpy as np
import torch
import random


VIDEO_SOFT_TOKEN = "<unused0>"
VIDEO_START_TOKEN = "<unused1>"


class VideoCollator:
    """
    collator for my mbart model, which changes lots of things from the original gfslt collator
    """

    def __init__(
        self,
        processor,
    ):
        super().__init__()
        self.processor = processor

    def __call__(self, batch):
        # Collate a batch of samples.
        zbatch = {key: tuple(dic[key] for dic in batch) for key in batch[0]}

        # Unpack batch data
        names, videos, texts = (
            zbatch["id"],
            zbatch["video"],
            zbatch["text"],
        )

        # Stack all videos into single tensor
        # video (T, C, H, W) ...
        #
        batch_features = self.processor(videos=videos)

        return {
            "pixel_values": batch_features.pixel_values,  # (B, C, H, W)
            "pixel_values_length": batch_features.pixel_values_lengths,  # (B, )
            # other useful info
            "names": names,
            "target_text": texts,
        }
