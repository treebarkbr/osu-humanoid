"""Human noise layer.

Turns the nets' *clean* output into a human-looking play:

  * tapping  -> hit-error jitter whose spread is the player's unstable rate,
                plus optional bias and occasional lapses; gets worse as local
                object density rises (choking).
  * aim      -> per-frame cursor tremor plus a touch of overshoot on big jumps.

The same module is used to (a) augment training labels so the nets learn to
reproduce human variability and (b) perturb a clean rollout at inference.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HumanParams:
    ur: float = 150.0          # unstable rate (10 * std of hit errors, ms)
    bias_ms: float = 0.0       # systematic early(-)/late(+) bias
    jitter_px: float = 2.0     # per-frame cursor tremor (arch.)-ish, px
    overshoot: float = 0.04    # extra travel on large jumps
    lapse_rate: float = 0.008  # chance per object of a big slip
    lapse_ms: float = 55.0     # extra sigma added on a lapse
    choke_gain: float = 0.6    # density -> sigma scaling
    density_ref: float = 4.0   # objects/sec where choke factor == 1

    @property
    def sigma(self) -> float:
        return self.ur / 10.0


PROFILES = {
    "pro":     HumanParams(ur=90.0, jitter_px=1.2, lapse_rate=0.004),
    "good":    HumanParams(ur=130.0, jitter_px=1.8),
    "average": HumanParams(ur=180.0, jitter_px=2.6, lapse_rate=0.012),
    "casual":  HumanParams(ur=260.0, jitter_px=4.0, lapse_rate=0.02, lapse_ms=70.0),
}


def from_profile(name: str) -> HumanParams:
    p = PROFILES.get(name)
    if p is None:
        raise KeyError(f"unknown profile {name!r}; have {sorted(PROFILES)}")
    return HumanParams(**vars(p))


def estimate_params_from_errors(errors: np.ndarray) -> HumanParams:
    """Recover a HumanParams from a real replay's signed hit errors (ms)."""
    e = np.asarray(errors, dtype=np.float64)
    if e.size < 2:
        return HumanParams()
    return HumanParams(ur=float(10.0 * np.std(e)), bias_ms=float(np.mean(e)))


def timing_sigma(params: HumanParams, density: float) -> float:
    """Per-object timing sigma, widened by choking at high local density."""
    choke = 1.0 + params.choke_gain * max(0.0, density / params.density_ref - 1.0)
    return params.sigma * choke


def sample_error(rng: np.random.Generator, params: HumanParams, density: float = 0.0) -> float:
    err = params.bias_ms + rng.normal(0.0, timing_sigma(params, density))
    if rng.random() < params.lapse_rate:
        err += rng.normal(0.0, params.lapse_ms)
    return float(err)


def humanize_taps(nominal_times, params: HumanParams, densities=None, *, rng=None):
    """nominal_times: press times (ms). densities: objects/sec per press."""
    rng = rng or np.random.default_rng()
    nominal = np.asarray(nominal_times, dtype=np.float64)
    dens = np.zeros(len(nominal)) if densities is None else np.asarray(densities, dtype=np.float64)
    errs = np.array([sample_error(rng, params, d) for d in dens])
    return nominal + errs, errs


def jitter_cursor(x: np.ndarray, y: np.ndarray, speed: np.ndarray,
                  params: HumanParams, *, rng=None):
    """Add tremor; faster cursor -> slightly larger, smoothed noise."""
    rng = rng or np.random.default_rng()
    n = len(x)
    nz = rng.normal(0.0, 1.0, size=(n, 2))
    amp = params.jitter_px * (0.5 + 0.5 * np.clip(speed / 3.0, 0.0, 2.0))
    dx = np.convolve(nz[:, 0] * amp, np.ones(3) / 3.0, mode="same")
    dy = np.convolve(nz[:, 1] * amp, np.ones(3) / 3.0, mode="same")
    return x + dx, y + dy


def overshoot_path(pred_delta: np.ndarray, params: HumanParams, *, rng=None) -> np.ndarray:
    """Scale large cursor deltas up slightly (people overshoot long jumps)."""
    rng = rng or np.random.default_rng()
    d = np.asarray(pred_delta, dtype=np.float64)
    mag = np.hypot(d[..., 0], d[..., 1])
    gain = 1.0 + params.overshoot * np.clip(mag / 0.15, 0.0, 1.5)
    gain += rng.normal(0.0, params.overshoot * 0.25, size=gain.shape)
    return d * gain[..., None]
