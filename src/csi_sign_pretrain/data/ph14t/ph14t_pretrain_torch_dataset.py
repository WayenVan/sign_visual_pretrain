import random
from typing import List, Tuple
import os
from datasets import load_dataset
import numpy
import pyspng


def sample_anchor_with_positives_in_segment(
    length: int,
    K: int,
    segment_index: int,
    N: int = 2,
    min_offset: int = 5,
    max_offset: int = 30,
    jitter: int = 0,
) -> Tuple[int, List[int]]:
    """
    从视频的某个指定段落采样一个锚点，并抽取正样本

    Args:
        length (int): 视频总帧数
        K (int): 视频划分的段数
        segment_index (int): 指定要采样的段落索引 (0 <= segment_index < K)
        N (int): 每个锚点对应的正样本数量
        min_offset (int): 正样本与锚点的最小间隔（帧数）
        max_offset (int): 正样本与锚点的最大间隔（帧数）
        jitter (int): 锚点采样的段内抖动范围

    Returns:
        Tuple[int, List[int]]:
            (anchor_index, [pos_index1, pos_index2, ...])
    """
    assert 0 <= segment_index < K, f"segment_index 必须在 [0, {K - 1}]"

    if length <= 0:
        return -1, []

    # ---- step1: 计算段落的范围 ----
    seg_len = length / K
    start = int(round(segment_index * seg_len))
    end = int(round((segment_index + 1) * seg_len))

    if end <= start:
        anchor = start
    else:
        anchor = random.randint(start, end - 1)

    # jitter
    if jitter > 0:
        low = max(0, anchor - jitter)
        high = min(length - 1, anchor + jitter)
        anchor = random.randint(low, high)

    # ---- step2: 对该 anchor 抽取 N 个正样本 ----
    positives = []
    for _ in range(N):
        candidates = []
        left_min = max(0, anchor - max_offset)
        left_max = max(0, anchor - min_offset)
        right_min = min(length - 1, anchor + min_offset)
        right_max = min(length - 1, anchor + max_offset)

        if left_max >= left_min:
            candidates.extend(range(left_min, left_max + 1))
        if right_max >= right_min:
            candidates.extend(range(right_min, right_max + 1))

        candidates = [c for c in candidates if c != anchor]

        if len(candidates) == 0:
            pos = anchor  # fallback
        else:
            pos = random.choice(candidates)
        positives.append(pos)

    return anchor, positives


class Ph14TPretrainTorchDataset:
    def __init__(
        self,
        data_root: str,
        mode: str = "train",
        pipline=None,
        K: int = 6,
        min_offset: int = 2,
        max_offset: int = 6,
        jitter: int = 0,
    ):
        self.K = K
        self.min_offset = min_offset
        self.max_offset = max_offset
        self.jitter = jitter

        self.data_root = data_root
        self.hg_dataset = load_dataset(
            "WayenVan/PHOENIX-Weather14T",
            split=mode,
            name="video_level",
        )
        # self.df = self.hg_dataset.to_polars()
        self.mode = mode

        self.pipline = pipline

    def __len__(self):
        return len(self.ids) * self.K

    @property
    def ids(self):
        return self.hg_dataset.unique("name")

    def __getitem__(self, idx):
        segment_index = idx % self.K
        video_index = idx // self.K

        if video_index >= len(self.ids):
            raise IndexError("Index out of range")

        id = self.ids[video_index]
        data_info = self.hg_dataset[video_index]

        assert id == data_info["name"], "ID is not consistent."

        video_frame_file_name = data_info["frames"]
        video_length = len(video_frame_file_name)

        anchor, positives = sample_anchor_with_positives_in_segment(
            length=video_length,
            K=self.K,
            segment_index=segment_index,
            N=1,
            min_offset=self.min_offset,
            max_offset=self.max_offset,
            jitter=self.jitter,
        )
        positive = positives[0]
        if anchor == -1:
            raise ValueError(f"Video {id} has no frames.")

        anchor_frame_file = video_frame_file_name[anchor]
        positive_frame_file = video_frame_file_name[positive]

        # anchor_frame = cv2.imread(os.path.join(self.data_root, anchor_frame_file))
        # anchor_frame = cv2.cvtColor(anchor_frame, cv2.COLOR_BGR2RGB)
        # positive_frame = cv2.imread(os.path.join(self.data_root, positive_frame_file))
        # positive_frame = cv2.cvtColor(positive_frame, cv2.COLOR_BGR2RGB)
        anchor_frame = self.read_png(os.path.join(self.data_root, anchor_frame_file))
        positive_frame = self.read_png(
            os.path.join(self.data_root, positive_frame_file)
        )

        ret = dict(
            id=id,
            segment_index=segment_index,
            # NOTE: [time, height, width, channel], normalized to [0, 1]
            anchor=numpy.array(anchor_frame, dtype=numpy.float32) / 255.0,
            positive=numpy.array(positive_frame, dtype=numpy.float32) / 255.0,
        )

        if self.pipline:
            ret = self.pipline(ret)

        return ret

    @staticmethod
    def read_png(file_name: str):
        with open(file_name, "rb") as f:
            image = pyspng.load(f.read())
        image = image[:, :, :3]
        return image


if __name__ == "__main__":
    import sys

    sys.path.append("./src")
    from csi_sign_pretrain.data.piplines.pretrain_pipline import PretrainPipline

    data_root = "dataset/PHOENIX-2014-T-release-v3/"
    ph14t_dataset = Ph14TPretrainTorchDataset(
        data_root, mode="train", pipline=PretrainPipline()
    )
    print(f"Dataset size: {len(ph14t_dataset)}")
    for i in range(10):
        data = ph14t_dataset[i]
        print(f"Data {i}:")
        for k, v in data.items():
            if isinstance(v, numpy.ndarray) or hasattr(v, "shape"):
                print(f"  {k}: {v.shape}, {v.dtype}, min={v.min()}, max={v.max()}")
            else:
                print(f"  {k}: {v}")
