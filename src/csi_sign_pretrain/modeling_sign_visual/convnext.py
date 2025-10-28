from torch import nn
import torch
import timm
from timm.models.convnext import ConvNeXt


import logging

logger = logging.getLogger(__name__)


class ConvNextBackbone(nn.Module):
    def __init__(self, id="convnext_tiny", pretrained: bool = True):
        super().__init__()
        self.model: ConvNeXt = timm.create_model(id, pretrained=pretrained)

    def forward(self, x: torch.Tensor):
        feats = self.model.forward_features(x)
        feats = feats.mean(dim=(2, 3)).contiguous()  # Global average pooling
        return feats


if __name__ == "__main__":
    model = ConvNextBackbone()
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(out.shape)
