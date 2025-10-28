from torch import nn
import torch

from transformers.models.resnet.modeling_resnet import ResNetForImageClassification


import logging

logger = logging.getLogger(__name__)


class ResnetBackbone(nn.Module):
    def __init__(self, id="microsoft/resnet-18"):
        super().__init__()
        self.model = ResNetForImageClassification.from_pretrained(id).resnet

    def forward(self, x: torch.Tensor):
        feats = self.model(x).pooler_output.flatten(start_dim=1)
        return feats


if __name__ == "__main__":
    model = ResnetBackbone()
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(out.shape)
