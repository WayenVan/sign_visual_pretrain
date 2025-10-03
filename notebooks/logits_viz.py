import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")

with app.setup:
    import os
    import marimo as mo
    import hydra
    import sys
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import torch

    sys.path.append("../src")
    from csi_sign_pretrain.data.datamodule import DataModule
    from csi_sign_pretrain.modeling_sign_visual.sign_pt_model import (
        SignVisualModelForPretrain,
    )
    import torch.nn.functional as F

    CHECKPOINT_PATH = "/root/projects/sign_langauge_visual_pretrain/outputs/first_demo_ft/2025-10-03_19-58-51/checkpoint-400"
    BATCH_SIZE = 8
    print("Current working directory:", os.getcwd())


@app.cell
def _():
    with hydra.initialize(
        config_path="../../root/projects/sign_langauge_visual_pretrain/configs"
    ):
        cfg = hydra.compose(config_name="base_train")
        cfg.data.data_root = "/root/projects/sign_langauge_visual_pretrain/dataset/PHOENIX-2014-T-release-v3"

    datamodule = DataModule(
        cfg.data,
    )
    datamodule.setup("train")
    train_dataset = datamodule.train_dataset

    loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=datamodule.collator,
    )
    return (loader,)


@app.cell
def _(loader):
    model = SignVisualModelForPretrain.from_pretrained(CHECKPOINT_PATH)

    with torch.no_grad():
        for batch in loader:
            anchors = batch["anchors"]
            positives = batch["positives"]
            student_feats = model(anchors).projection_feats
            teacher_feats = model(positives).projection_feats
            break
    print("Anchor features shape:", student_feats.shape)
    print("Positive features shape:", teacher_feats.shape)
    return student_feats, teacher_feats


@app.cell
def visualize_logits(student_feats, teacher_feats):
    student_prob = F.softmax(student_feats, dim=-1)
    teacher_prob = F.softmax(teacher_feats, dim=-1)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Student Probabilities")
    plt.imshow(student_prob.numpy(), aspect="auto", cmap="hot")

    plt.subplot(1, 2, 2)
    plt.title("Teacher Probabilities")
    plt.imshow(teacher_prob.numpy(), aspect="auto", cmap="hot")

    plt.colorbar()
    plt.show()
    return


@app.function
def tensor_to_image(image):
    image = image.clone()
    image = image.permute(0, 2, 3, 1)  # C, H, W to H, W, C
    image = image * 0.5 + 0.5  # assuming mean=0.5, std=0.5
    image = image.clamp(0, 1)
    image = image.numpy()
    # for i in range(image.shape[0]):
    #     image[i] = cv2.cvtColor(image[i], cv2.COLOR_RGB2BGR)
    # image = image[..., ::-1]  # RGB to BGR
    return image


if __name__ == "__main__":
    app.run()
