import torch
from transformers.modeling_utils import PreTrainedModel
import torch.nn.functional as F
from .ouput_utils import SignPretrainOutput
from .registry import VISUAL_BACKBONES
from ..configuration_sign_visual.configuration import SignPretrainConfig


class SignVisualModelForPretrain(PreTrainedModel):
    config_class = SignPretrainConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.backbone = VISUAL_BACKBONES[config.backbone_type](**config.backbone_kwargs)
        self.register_buffer(
            "center", torch.zeros(1, config.projection_size)
        )  # 初始化中心向量
        self.proj = torch.nn.Linear(config.hidden_size, config.projection_size)

    def forward(self, pixel_values, return_dict=False, projection=True, **kwargs):
        feats = self.backbone(pixel_values)

        projection_feats = self.proj(feats) if projection else None

        if not return_dict:
            return SignPretrainOutput(feats=feats, projection_feats=projection_feats)
        return dict(feats=feats, projection_feats=projection_feats)

    @property
    def dummy_inputs(self):
        return {"pixel_values": torch.zeros(1, 3, 224, 224).to(self.device)}
