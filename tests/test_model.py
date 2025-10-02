import sys

sys.path.append("./src")
from csi_sign_pretrain.configuration_sign_visual.configuration import SignPretrainConfig
from csi_sign_pretrain.modeling_sign_visual.sign_pt_model import (
    SignVisualModelForPretrain,
)


def test_sign_pretrain_model():
    config = SignPretrainConfig(
        hidden_size=768,
        backbone_type="dinov2-with-token-sampler",
        backbone_kwargs=dict(
            id="facebook/dinov2-with-registers-base",
            output_layer=-2,
            num_extra_queries=4,
            num_hdeads=12,
            num_adapter_layers=2,
        ),
    )
    model = SignVisualModelForPretrain(config)
    model.save_pretrained("outputs/test-sign-pretrain-model")


if __name__ == "__main__":
    test_sign_pretrain_model()
