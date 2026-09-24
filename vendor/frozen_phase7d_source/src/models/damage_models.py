"""
DamageACT-U pilot models.

Two capacity-matched models are intentionally provided:

1) post_only
   - receives only POST-event visual information.
   - the POST feature is repeated three times before the shared classifier head.
   - this adds no new information but preserves the same 1536->512->4 head and
     therefore the same trainable parameter count as the paired model.

2) siamese
   - uses a shared ResNet-18 encoder:
       z_pre  = E(pre)
       z_post = E(post)
       d      = |z_post - z_pre|
       h      = [z_pre, z_post, d]

The capacity-matched design makes the PRE-vs-no-PRE comparison cleaner:
both models have the same trainable architecture/parameter count; only the
information supplied to the fusion representation differs.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


CLASS_NAMES = ["no-damage", "minor-damage", "major-damage", "destroyed"]


class SharedResNet18Encoder(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        self.feature_dim = int(backbone.fc.in_features)  # 512
        backbone.fc = nn.Identity()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class DamageACTClassifier(nn.Module):
    """
    model_type:
        - "post_only"
        - "siamese"
    """

    def __init__(
        self,
        model_type: str,
        num_classes: int = 4,
        hidden_dim: int = 512,
        dropout: float = 0.30,
        pretrained: bool = True,
    ):
        super().__init__()

        if model_type not in {"post_only", "siamese"}:
            raise ValueError("model_type must be 'post_only' or 'siamese'")

        self.model_type = model_type
        self.encoder = SharedResNet18Encoder(pretrained=pretrained)
        d = self.encoder.feature_dim
        self.fusion_dim = 3 * d

        self.head = nn.Sequential(
            nn.Linear(self.fusion_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(hidden_dim, num_classes),
        )

    def make_fused_features(
        self,
        pre: torch.Tensor,
        post: torch.Tensor,
    ) -> torch.Tensor:
        z_post = self.encoder(post)

        if self.model_type == "post_only":
            # Capacity-matched control: three copies contain no additional
            # information beyond POST, while the classifier head remains
            # exactly the same size as in the paired model.
            return torch.cat([z_post, z_post, z_post], dim=1)

        z_pre = self.encoder(pre)
        delta = torch.abs(z_post - z_pre)
        return torch.cat([z_pre, z_post, delta], dim=1)

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        h = self.make_fused_features(pre, post)
        return self.head(h)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
