from transformers.configuration_utils import PretrainedConfig
from typing import Any, Dict, Optional


class SignPretrainConfig(PretrainedConfig):
    """Configuration class for SLT (Sign Language Translation) model.

    This class stores the configuration for the SLT model, including settings for
    the language model, visual backbone, and adapters.
    """

    model_type = "slt"

    def __init__(
        self,
        hidden_size: int = 768,
        backbone_type: str = "dinov2-with-token-sampler",
        backbone_kwargs: Optional[Dict[str, Any]] = None,
        projection_size: int = 5120,
        **kwargs: Any,
    ):
        """
        Initialize SignPretrainConfig.
        Args:
            hidden_size: Dimensionality of the model's hidden states.
            backbone_type: Type of visual backbone to use. Options: 'resnet50', 'vit-base-patch16-224', etc.
            backbone_kwargs: Additional keyword arguments for the visual backbone model.
            projection_size: Dimensionality of the projection layer output.
            **kwargs: Additional arguments passed to the parent PretrainedConfig class.
        """
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.backbone_type = backbone_type
        self.backbone_kwargs = backbone_kwargs if backbone_kwargs is not None else {}
        self.projection_size = projection_size
