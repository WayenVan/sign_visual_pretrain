import numpy as np
import torch
import random


VIDEO_SOFT_TOKEN = "<unused0>"
VIDEO_START_TOKEN = "<unused1>"


class VideoCollator:
    """
    collator for my mbart model, which changes lots of things from the original gfslt collator
    <unused0> is used as image soft token, <unused1> is used as image sentinel
    """

    def __init__(
        self,
    ):
        super().__init__()

    @staticmethod
    def pad_dim_to_multiple_of_4(tensor, dim):
        current_size = tensor.size(dim)
        remainder = current_size % 4
        if remainder == 0:
            return tensor

        pad_size = 4 - remainder

        # 取这个维度的最后一个元素
        index = [slice(None)] * tensor.dim()
        index[dim] = -1
        last_element = tensor[tuple(index)].unsqueeze(dim)

        # 复制 pad_size 次
        padding = last_element.repeat_interleave(pad_size, dim=dim)
        return torch.cat([tensor, padding], dim=dim).contiguous()

    def __call__(self, batch):
        # Collate a batch of samples.
        zbatch = {key: tuple(dic[key] for dic in batch) for key in batch[0]}

        # Unpack batch data
        names, videos, texts = (
            zbatch["id"],
            zbatch["augmented_video"],
            zbatch["text"],
        )

        # Stack all videos into single tensor
        # video (T, C, H, W) ...
        #
        videos = [self.pad_dim_to_multiple_of_4(video, dim=0) for video in videos]
        video_lengths = [video.size(0) for video in videos]

        video_tensor = torch.cat(videos, dim=0).contiguous()
        video_lengths_tensor = torch.tensor(video_lengths)

        return {
            "pixel_values": video_tensor,
            "pixel_values_length": video_lengths_tensor,
            # other useful info
            "names": names,
            "target_text": texts,
        }
