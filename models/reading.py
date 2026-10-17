"""Reading head: anticipates the cursor position at the next object's hit time.

Unlike the aim head this gets *no* cursor history -- only the upcoming pattern
(relative positions, timing, type) plus difficulty -- so it must learn the
human's placement/offset from the map alone.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from pipeline.torch_data import K, W


class ReadingNet(nn.Module):
    def __init__(self, w: int = W, k: int = K, hidden: int = 128):
        super().__init__()
        self.w, self.k = w, k
        self.net = nn.Sequential(
            nn.Linear(k * 4 + 4, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, hist: torch.Tensor, ctx: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        x = torch.cat([ctx.flatten(1), meta], dim=1)
        return self.net(x)
