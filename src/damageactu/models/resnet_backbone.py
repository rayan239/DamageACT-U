from __future__ import annotations


def make_reference_classifier(mode: str, num_classes: int = 4, pretrained: bool = True):
    import torch
    import torch.nn as nn
    from torchvision.models import resnet18, ResNet18_Weights


    if mode not in {"post_only","siamese"}:
        raise ValueError(mode)


    class ReferenceDamageACTClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            base = resnet18(weights=weights)
            self.encoder = nn.Sequential(*list(base.children())[:-1])
            self.head = nn.Sequential(
                nn.Linear(1536, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.30),
                nn.Linear(512, num_classes),
            )


        def encode(self, x):
            z = self.encoder(x)
            return torch.flatten(z, 1)


        def forward(self, pre, post=None):
            if mode == "post_only":
                xpost = pre if post is None else post
                zpost = self.encode(xpost)
                rep = torch.cat([zpost, zpost, zpost], dim=1)
            else:
                if post is None:
                    raise ValueError("Siamese mode requires PRE and POST tensors.")
                zpre = self.encode(pre)
                zpost = self.encode(post)
                rep = torch.cat([zpre, zpost, torch.abs(zpost-zpre)], dim=1)
            return self.head(rep)


    return ReferenceDamageACTClassifier()
