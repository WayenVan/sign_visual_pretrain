import hydra
import sys
import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer

sys.path.append("./src")
from csi_sign_pretrain.data.datamodule import DataModule


def test_datamodule():
    with hydra.initialize(config_path="../configs"):
        cfg = hydra.compose(config_name="base_train")

    datamodule = DataModule(
        cfg.data,
    )
    datamodule.setup("train")
    train_dataset = datamodule.train_dataset

    loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        collate_fn=datamodule.train_collator,
    )

    for batch in loader:
        # print(batch["input_ids"][0])
        # print(batch["labels"][0])
        print(batch["anchors"].shape)
        print(batch["positives"].shape)


def test_datamodule_video():
    with hydra.initialize(config_path="../configs"):
        cfg = hydra.compose(config_name="base_eval")

    datamodule = DataModule(
        cfg.data,
    )
    datamodule.setup("train")
    train_dataset = datamodule.train_dataset

    loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=datamodule.train_collator,
    )

    for batch in loader:
        # print(batch["input_ids"][0])
        # print(batch["labels"][0])
        print(batch["pixel_values"].shape)
        print(batch["pixel_values_length"])


if __name__ == "__main__":
    # test_datamodule()
    test_datamodule_video()
