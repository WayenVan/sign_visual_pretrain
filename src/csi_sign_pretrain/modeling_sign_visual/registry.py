from torch import nn
from typing import Dict, Type


from .dinov2_with_token_sampler import DinoV2WithTokenSampler
from .convnext import ConvNextBackbone
from .resnet import ResnetBackbone

VISUAL_BACKBONES: Dict[str, Type[nn.Module]] = {
    "dinov2-with-token-sampler": DinoV2WithTokenSampler,
    "convnext": ConvNextBackbone,
    "resnet": ResnetBackbone,
}
