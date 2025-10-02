from transformers.modeling_outputs import ModelOutput, CausalLMOutput
from dataclasses import dataclass

import torch


@dataclass
class SignPretrainOutput(ModelOutput):
    feats: torch.FloatTensor
    unpooled_feats: torch.FloatTensor | None = None
    attentions: tuple[torch.FloatTensor] | None = None
