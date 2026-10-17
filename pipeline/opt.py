"""Training utilities ported from the RoNet recipes (Luau) to PyTorch.

RoNet is a from-scratch Luau autograd library; nothing imports directly, but
its optimizer/loss/schedule recipes port cleanly:

* ``MuonAdamW``  -- modded-nanogpt / Moonlight composite: 2D hidden matrices
  get Muon (SGD-momentum + Newton-Schulz orthogonalization, nesterov), every
  other parameter gets AdamW at its own lr.
* ``EMA``        -- Polyak averaging with the hana-rhea warmup decay
  ``min(decay, (1+t)/(t+2))``.
* ``warmup_cosine`` -- linear warmup then cosine decay to a floor.
* ``focal_bce``  -- focal loss (Lin et al. 2017) on raw logits for the
  imbalanced tapping head.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _newton_schulz(g: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Quintic Newton-Schulz iteration (modded-nanogpt coefficients)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g / (g.norm() + eps)
    tall = x.shape[0] > x.shape[1]
    if tall:
        x = x.T
    for _ in range(steps):
        aa = x @ x.T
        bb = b * aa + c * (aa @ aa)
        x = a * x + bb @ x
    if tall:
        x = x.T
    return x


class Muon:
    """Muon over 2D parameters: nesterov momentum then orthogonalized update."""

    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.0,
                 ns_steps=5):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.momentum = momentum
        self.wd = weight_decay
        self.ns_steps = ns_steps
        self.buf = [torch.zeros_like(p) for p in self.params]

    @torch.no_grad()
    def zero_grad(self):
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            buf = self.buf[i]
            buf.mul_(self.momentum).add_(g, alpha=1 - self.momentum)
            update = g.add(buf - g, alpha=self.momentum)
            orth = _newton_schulz(update, self.ns_steps)
            scale = max(1.0, p.shape[0] / p.shape[1]) ** 0.5
            if self.wd:
                p.mul_(1 - self.lr * self.wd)
            p.add_(orth, alpha=-self.lr * scale)

    def set_lr(self, lr):
        self.lr = lr


def _split_params(model: nn.Module):
    """2D weight matrices -> Muon; everything else -> AdamW."""
    muon, adam = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and name.endswith("weight"):
            muon.append(p)
        else:
            adam.append(p)
    return muon, adam


class MuonAdamW:
    """Composite two-optimizer step so both share one gradient pass."""

    def __init__(self, model: nn.Module, *, lr=1e-3, muon_lr=0.02,
                 weight_decay=1e-2, momentum=0.95, ns_steps=5):
        muon, adam = _split_params(model)
        self.muon = Muon(muon, lr=muon_lr, momentum=momentum,
                         weight_decay=weight_decay, ns_steps=ns_steps) if muon else None
        self.adam = torch.optim.AdamW(adam, lr=lr, weight_decay=weight_decay) if adam else None
        self._muon_base = muon_lr
        self._adam_base = lr

    def zero_grad(self):
        if self.muon:
            self.muon.zero_grad()
        if self.adam:
            self.adam.zero_grad()

    def step(self):
        if self.muon:
            self.muon.step()
        if self.adam:
            self.adam.step()

    def set_scale(self, factor: float):
        if self.muon:
            self.muon.set_lr(self._muon_base * factor)
        if self.adam:
            for grp in self.adam.param_groups:
                grp["lr"] = self._adam_base * factor


class EMA:
    """Polyak averaging with hana-rhea warmup decay. Use ``copy_to`` before eval."""

    def __init__(self, model: nn.Module, decay: float = 0.999, warmup: int = 0):
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.max_decay = decay
        self.warmup = warmup
        self.step_n = 0
        self.shadow = [p.detach().clone() for p in self.params]
        self._backup = None

    def _decay(self) -> float:
        t = self.step_n
        if self.warmup and t < self.warmup:
            return min(self.max_decay, (1 + t) / (t + 2))
        return self.max_decay

    @torch.no_grad()
    def update(self):
        self.step_n += 1
        d = 1 - self._decay()
        for s, p in zip(self.shadow, self.params):
            s.mul_(1 - d).add_(p.detach(), alpha=d)

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        self._backup = [p.detach().clone() for p in self.params]
        for s, p in zip(self.shadow, self.params):
            p.copy_(s)

    @torch.no_grad()
    def restore(self):
        if self._backup is None:
            return
        for b, p in zip(self._backup, self.params):
            p.copy_(b)
        self._backup = None


def warmup_cosine(warmup: int, total: int, min_factor: float = 0.05):
    """Linear warmup for [0, warmup], then cosine decay to ``min_factor``."""
    floor = min_factor
    amp = 1 - floor
    def f(step: int) -> float:
        if warmup and step < warmup:
            return step / warmup
        t = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
        return floor + amp * 0.5 * (1 + __import__("math").cos(3.141592653589793 * t))
    return f


def focal_bce(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0,
              alpha: float = 0.5, label_smooth: float = 0.0) -> torch.Tensor:
    """Binary focal loss (Lin et al. 2017) from raw logits, mean-reduced."""
    if label_smooth:
        targets = targets * (1 - label_smooth) + 0.5 * label_smooth
    logp = nn.functional.logsigmoid(logits)
    log1mp = nn.functional.logsigmoid(-logits)
    p = logp.exp()
    pt = p * targets + (1 - p) * (1 - targets)
    logpt = logp * targets + log1mp * (1 - targets)
    a = alpha * targets + (1 - alpha) * (1 - targets)
    loss = -a * (1 - pt).pow(gamma) * logpt
    return loss.mean()
