import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


with app.setup():
    import os
    import marimo as mo
    import hydra
    import sys
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import cv2

    sys.path.append("../src")
    from csi_sign_pretrain.data.datamodule import DataModule

    BATCH_SIZE = 2
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


@app.cell
def visualize_batch(loader):
    for batch in loader:
        anchors = batch["anchors"]
        positives = batch["positives"]
        break

    anchors = tensor_to_image(anchors)
    positives = tensor_to_image(positives)

    fig, axes = plt.subplots(BATCH_SIZE, 2, figsize=(16, 8))
    for b in range(BATCH_SIZE):
        axes[b, 0].imshow(anchors[b])
        axes[b, 0].axis("off")
        axes[b, 0].set_title("Anchor")

        axes[b, 1].imshow(positives[b])
        axes[b, 1].axis("off")
        axes[b, 1].set_title("Positive")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    app.run()
