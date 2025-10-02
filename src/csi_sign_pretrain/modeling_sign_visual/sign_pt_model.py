import torch
from transformers.modeling_utils import PreTrainedModel
import torch.nn.functional as F
from .ouput_utils import SignPretrainOutput
from .registry import VISUAL_BACKBONES


class SignVisualModelForPretrain(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.backbone = VISUAL_BACKBONES[config.backbone_type](**config.backbone_kwargs)
        self.center = self.register_buffer(
            "center", torch.zeros(1, config.hidden_size)
        )  # 初始化中心向量

    def forward(self, pixel_values, return_dict=False, **kwargs):
        feats = self.backbone(pixel_values)
        if not return_dict:
            return SignPretrainOutput(feats=feats)
        return dict(feats=feats)
