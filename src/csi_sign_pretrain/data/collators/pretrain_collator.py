import torch


class PretrainCollator:
    def __init__(self):
        super().__init__()

    def __call__(self, batch):
        # Collate a batch of samples.
        anchors = [item["anchor"] for item in batch]
        positives = [item["positive"] for item in batch]
        anchors = torch.stack(anchors, dim=0)
        positives = torch.stack(positives, dim=0)
        return {
            "anchors": anchors,
            "positives": positives,
        }
