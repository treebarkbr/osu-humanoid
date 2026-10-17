"""Aim head: predicts the cursor delta for the next frame from recent motion."""
from __future__ import annotations

import torch
import torch.nn as nn

from pipeline.torch_data import ACT, K, W


class AimNet(nn.Module):
    def __init__(self, w: int = W, k: int = K, act: int = ACT, hidden: int = 256):
        super().__init__()
        self.w, self.k, self.act = w, k, act
        self.net = nn.Sequential(
            nn.Linear(w * 5 + k * 4 + 4 + act, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, hist: torch.Tensor, ctx: torch.Tensor, meta: torch.Tensor,
                act: torch.Tensor | None = None) -> torch.Tensor:
        parts = [hist.flatten(1), ctx.flatten(1), meta]
        if act is not None and self.act > 0:
            parts.append(act)
        return self.net(torch.cat(parts, dim=1))
