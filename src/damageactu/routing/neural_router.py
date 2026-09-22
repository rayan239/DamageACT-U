from __future__ import annotations


def build_neural_router(n_features=26):
    import torch.nn as nn


    class TemporalUtilityRouter(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_features, 32),
                nn.ReLU(),
                nn.Dropout(0.10),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )
        def forward(self, x):
            return self.net(x).squeeze(1)


    return TemporalUtilityRouter()


def routed_logits_torch(post_logits, pair_logits, gate_logits):
    import torch
    u = torch.sigmoid(gate_logits).unsqueeze(1)
    return post_logits + u * (pair_logits-post_logits), u.squeeze(1)
