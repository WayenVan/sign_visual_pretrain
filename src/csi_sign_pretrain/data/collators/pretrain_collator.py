import torch
import numpy as np


class PretrainCollator:
    def __init__(self, processor, debug=False):
        super().__init__()
        self.processor = processor
        self.debug = False

    def __call__(self, batch):
        # Collate a batch of samples.
        anchors = [item["anchor"] for item in batch]
        positives = [item["positive"] for item in batch]
        anchors = np.stack(anchors, axis=0)
        positives = np.stack(positives, axis=0)

        batched_features = self.processor(anchors=anchors, positives=positives)

        ret = {
            **batched_features,
        }
        if self.debug:
            ret["original_anchors"] = anchors
            ret["original_positives"] = positives

        return ret
