"""Composite reward from a danser-stable JudgeReport (scheme B).

The reward is a *smooth surrogate of osu!stable accuracy*: the per-object
`base` term sums to the official accuracy (300->1, 100->1/3, 50->1/6, miss->0),
while dense shaping terms reward the same continuous quantities the accuracy
step-function thresholds (tap timing error, aim distance, slider tracking,
spinner rate). This gives a usable gradient for near-perfect play without
changing what "good" means. A small combo term reflects that ScoreV1 is
combo-multiplied.

    R_map = mean_o[ base + w_time*timing + w_track*track + w_aim*aim + w_spin*spin ]
          + w_combo * (max_combo / N)

Every term is per-object and normalised, so long/difficult maps don't dominate.
"""
from __future__ import annotations

from .hitwindows import difficulty_range, object_radius
from .judge import J300, J100, J50, JMISS

# Balanced weights (accuracy dominant). See design notes.
WEIGHTS = {
    "time": 0.15,     # tap timing precision (circles + slider heads)
    "track": 0.20,    # slider tracking (== danser's pre-threshold `rate`)
    "aim": 0.10,      # cursor proximity at press
    "spin": 0.10,     # spinner rotation fraction
    "combo": 0.15,    # consistency (combo-multiplied scoring)
}

_BASE = {J300: 1.0, J100: 1.0 / 3.0, J50: 1.0 / 6.0, JMISS: 0.0}

_HIT50_RANGE = (200.0, 150.0, 100.0)  # OD -> 50-window half width


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def reward_map(rep, *, radius: float, od: float, w: dict | None = None) -> tuple[float, int]:
    """Return (reward, max_combo) for one judged map."""
    w = w or WEIGHTS
    hit50 = difficulty_range(od, _HIT50_RANGE)
    if hit50 <= 0:
        hit50 = 1.0
    n = len(rep.results)
    if n == 0:
        return 0.0, 0

    total = 0.0
    combo = 0
    max_combo = 0
    for r in rep.results:
        total += _BASE.get(r.kind, 0.0)

        if r.is_spinner:
            if r.spin_req > 0:
                total += w["spin"] * _clamp01(r.spin_count / r.spin_req)
            hit = r.kind != JMISS
        else:
            if r.is_slider:
                # tick_hits already includes the head; +1 in the denominator
                # makes this exactly danser's pre-threshold `rate`.
                total += w["track"] * _clamp01(r.tick_hits / (r.tick_total + 1.0))
            if r.hit_error is not None:
                total += w["time"] * _clamp01(1.0 - abs(r.hit_error) / hit50)
            if r.dist is not None and radius > 0:
                total += w["aim"] * _clamp01(1.0 - r.dist / radius)
            hit = r.kind != JMISS
            if r.is_slider and r.dropped:
                hit = False  # a tick break resets combo, like osu!
        if hit:
            combo += 1
            if combo > max_combo:
                max_combo = combo
        else:
            combo = 0

    reward = total / n + w["combo"] * (max_combo / n)
    return reward, max_combo


def radius_for(bm, mods=()) -> float:
    """Object radius for the map under the given mods (default: no mods)."""
    from .mods import effective_difficulty
    cs = effective_difficulty(bm.cs, bm.ar, bm.od, bm.hp, mods)[0]
    return object_radius(cs)


def od_for(bm, mods=()) -> float:
    from .mods import effective_difficulty
    return effective_difficulty(bm.cs, bm.ar, bm.od, bm.hp, mods)[2]
