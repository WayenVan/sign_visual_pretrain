from transformers.models.dinov2_with_registers.modeling_dinov2_with_registers import (
    Dinov2WithRegistersModel,
)
from torch import nn
import torch

from timm.models.vision_transformer import (
    DropPath,
    Mlp,
    LayerScale,
)


from typing import Optional
from transformers.models.gemma3.modeling_gemma3 import Gemma3RMSNorm
import logging
from einops import rearrange, repeat

logger = logging.getLogger(__name__)


def build_mlp(depth, hidden_size, output_hidden_size):
    modules = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        modules.append(nn.GELU())
        modules.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*modules)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm0 = norm_layer(dim)
        self.norm1 = norm_layer(dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            bias=True,
            batch_first=True,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        keys = self.norm0(keys)
        x = queries + self.drop_path1(
            self.ls1(self.attn(self.norm1(queries), keys, keys)[0])
        )
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class TokenSampleAdapter(nn.Module):
    def __init__(
        self,
        hidden_size,
        target_hidden_size,
        num_heads,
        num_layers,
        num_extra_queries,
        mlp_depth=1,
        mlp_ratio=2.0,
        proj_drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        eps=1e-6,
    ):
        super().__init__()
        self.num_extra_queries = num_extra_queries
        self.extra_queries = nn.Parameter(
            torch.randn(1, num_extra_queries, hidden_size), requires_grad=True
        )
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    proj_drop=proj_drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path,
                    act_layer=nn.GELU,
                    norm_layer=nn.LayerNorm,
                    mlp_layer=Mlp,
                )
                for _ in range(num_layers)
            ]
        )
        self.mlp = build_mlp(
            mlp_depth, hidden_size * self.num_extra_queries, target_hidden_size
        )
        self.norm = Gemma3RMSNorm(hidden_size, eps=eps)
        # self.positional_embedding = nn.Embedding(max_length, target_hidden_size)

    def forward(self, x):
        # x: (B, T, HW, C)
        BT, HW, C = x.shape

        extra_queries = repeat(self.extra_queries, "1 n c -> bt n c", bt=BT)
        for block in self.blocks:
            extra_queries = block(extra_queries, x)

        extra_queries = self.norm(
            extra_queries
        )  # (B*T, num_extra_queries, hidden_size)

        extra_queries = rearrange(
            extra_queries, "bt n c -> bt (n c)"
        )  # (B, T, num_extra_queries * hidden_size)
        feats = self.mlp(extra_queries)  # (B, T, Target_hidden_size)

        return feats


class DinoV2WithTokenSampler(nn.Module):
    def __init__(
        self,
        id,
        output_layer=-1,
        num_extra_queries=4,
        num_heads=12,
        num_adapter_layers=2,
    ):
        super().__init__()
        self.id = id
        self.visual_encoder = Dinov2WithRegistersModel.from_pretrained(id)
        self.token_sampler = TokenSampleAdapter(
            hidden_size=self.visual_encoder.config.hidden_size,
            target_hidden_size=self.visual_encoder.config.hidden_size,
            num_heads=num_heads,
            num_layers=num_adapter_layers,
            num_extra_queries=num_extra_queries,
            mlp_depth=1,
            mlp_ratio=2.0,
            proj_drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            eps=1e-6,
        )
        self.output_layer = output_layer
        self._post_init()

    def _post_init(self):
        for param in self.visual_encoder.parameters():
            param.requires_grad = False

    def forward(self, x) -> torch.Tensor:
        """
        [B, C, H, W]
        """
        B, C, H, W = x.shape
        hidden_states = self.visual_encoder(x, output_hidden_states=True).hidden_states

        return self.token_sampler(hidden_states[self.output_layer])


if __name__ == "__main__":
    model = DinoV2WithTokenSampler(
        id="facebook/dinov2-with-registers-base",
        output_layer=-2,
        num_extra_queries=4,
        num_hdeads=12,
        num_adapter_layers=2,
    )
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        feats = model(x)
    print(feats.shape)  # [B, T, C]l
