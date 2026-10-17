"""Hit windows + legacy ruleset helpers.

Sources:
  osu.Game.Rulesets.Osu/Scoring/OsuHitWindows.cs
  osu.Game/Rulesets/Objects/Legacy/LegacyRulesetExtensions.cs
"""
from __future__ import annotations

import math

MISS_WINDOW = 400.0

GREAT_WINDOW_RANGE = (80.0, 50.0, 20.0)
OK_WINDOW_RANGE = (140.0, 100.0, 60.0)
MEH_WINDOW_RANGE = (200.0, 150.0, 100.0)


def difficulty_range(difficulty: float, rng: tuple[float, float, float]) -> float:
    """IBeatmapDifficultyInfo.DifficultyRange."""
    min_v, mid_v, max_v = rng
    if difficulty > 5:
        return mid_v + (max_v - mid_v) * (difficulty - 5) / 5
    if difficulty < 5:
        return mid_v - (mid_v - min_v) * (5 - difficulty) / 5
    return mid_v


def osu_hit_windows(od: float) -> tuple[float, float, float, float]:
    """(great, ok, meh, miss) half-widths in ms."""
    great = math.floor(difficulty_range(od, GREAT_WINDOW_RANGE)) - 0.5
    ok = math.floor(difficulty_range(od, OK_WINDOW_RANGE)) - 0.5
    meh = math.floor(difficulty_range(od, MEH_WINDOW_RANGE)) - 0.5
    return great, ok, meh, MISS_WINDOW


def calculate_scale_from_circle_size(cs: float, apply_fudge: bool = False) -> float:
    """Radius scale (osu!std). object_radius = 64 * scale.

    The inner call is the single-arg DifficultyRange(cs) == (cs - 5) / 5.
    """
    broken_gamefield_rounding_allowance = 1.00041
    scale = (1.0 - 0.7 * ((cs - 5) / 5)) / 2
    if apply_fudge:
        scale *= broken_gamefield_rounding_allowance
    return scale


def object_radius(cs: float) -> float:
    return 64.0 * calculate_scale_from_circle_size(cs, apply_fudge=True)


def precision_adjusted_beat_length(beat_length: float, slider_velocity_multiplier: float) -> float:
    """osu! ruleset only."""
    sv_as_beat_length = -100.0 / slider_velocity_multiplier
    if sv_as_beat_length < 0:
        bpm_multiplier = min(max(-sv_as_beat_length, 10.0), 1000.0) / 100.0
    else:
        bpm_multiplier = 1.0
    return beat_length * bpm_multiplier


def approach_time(ar: float) -> float:
    if ar <= 5:
        return 1200 + 600 * (5 - ar) / 5
    return 1200 - 750 * (ar - 5) / 5
