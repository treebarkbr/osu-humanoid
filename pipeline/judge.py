"""Headless osu!standard judge modelled on the lazer ruleset.

Covers:
  * circles + slider heads  -> normal circle hit windows (OsuHitWindows)
  * slider ticks/repeats/tail -> SliderInputManager tracking model
  * spinners -> rotation-based completion

Mods are applied to the judged playfield: DT/NC (1.5x) and HT (0.75x) scale
object/event times; HR flips vertically and raises CS/AR/OD/HP; EZ lowers them.
Hit-window widths (from OD) stay in real time, as in the game.

Sources:
  osu.Game.Rulesets.Osu/Objects/Drawables/SliderInputManager.cs
  osu.Game.Rulesets.Osu/Objects/Drawables/DrawableSliderBall.cs (FOLLOW_AREA)
  osu.Game/Rulesets/Objects/SliderEventGenerator.cs (TAIL_LENIENCY)
  osu.Game.Rulesets.Osu/Objects/Drawables/DrawableSpinner.cs
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import slider_events as SE
from .beatmap import Beatmap, HitObject, curve_position, slider_events
from .hitwindows import difficulty_range, object_radius, osu_hit_windows
from .mods import clock_rate, effective_difficulty, is_hr
from .replay import ReplayFrames

J300, J100, J50, JMISS = "300", "100", "50", "miss"

FOLLOW_AREA = 2.4
TAIL_LENIENCY = SE.TAIL_LENIENCY
PLAYFIELD_H = 384.0

_LEFT_BITS = 0b0101   # M1 | K1
_RIGHT_BITS = 0b1010  # M2 | K2


@dataclass
class JudgementResult:
    obj_time: int
    kind: str
    hit_error: float | None
    dist: float | None
    press_time: int | None
    is_slider: bool = False
    is_spinner: bool = False
    dropped: bool = False
    tick_hits: int = 0
    tick_total: int = 0
    tick_misses: int = 0


@dataclass
class JudgeReport:
    results: list[JudgementResult] = field(default_factory=list)

    @property
    def hit_errors(self) -> np.ndarray:
        return np.array([r.hit_error for r in self.results if r.hit_error is not None],
                        dtype=np.float64)

    @property
    def drops(self) -> int:
        return sum(1 for r in self.results if r.dropped)

    @property
    def tick_misses(self) -> int:
        return sum(r.tick_misses for r in self.results)

    @property
    def tick_hits(self) -> int:
        return sum(r.tick_hits for r in self.results)

    @property
    def unstable_rate(self) -> float:
        e = self.hit_errors
        return float(10.0 * np.std(e)) if len(e) >= 2 else 0.0

    def counts(self) -> dict[str, int]:
        out = {J300: 0, J100: 0, J50: 0, JMISS: 0}
        for r in self.results:
            out[r.kind] += 1
        return out

    @property
    def accuracy(self) -> float:
        c = self.counts()
        total = c[J300] + c[J100] + c[J50] + c[JMISS]
        if total == 0:
            return 1.0
        return (300 * c[J300] + 100 * c[J100] + 50 * c[J50]) / (300 * total)

    def summary(self) -> str:
        c = self.counts()
        e = self.hit_errors
        mean = float(np.mean(e)) if len(e) else 0.0
        return (f"300s={c[J300]} 100s={c[J100]} 50s={c[J50]} miss={c[JMISS]} "
                f"drops={self.drops} tickmiss={self.tick_misses}/{self.tick_hits + self.tick_misses} "
                f"acc={self.accuracy * 100:.2f}% UR={self.unstable_rate:.1f} mean_err={mean:+.1f}ms")


# ---------------------------------------------------------------------------
def cursor_at(frames: ReplayFrames, t: float) -> tuple[float, float]:
    idx = int(np.searchsorted(frames.times, t, side="right")) - 1
    idx = max(0, idx)
    return float(frames.x[idx]), float(frames.y[idx])


def _actions(keys: int) -> set[str]:
    a: set[str] = set()
    if keys & _LEFT_BITS:
        a.add("L")
    if keys & _RIGHT_BITS:
        a.add("R")
    return a


def _head_action(mask: int) -> str | None:
    if mask & _LEFT_BITS:
        return "L"
    if mask & _RIGHT_BITS:
        return "R"
    return None


def press_events(frames: ReplayFrames) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(frame_index, time, newly_pressed_mask) for ascending per-button edges."""
    keys = frames.keys.astype(np.int32)
    prev = np.concatenate(([0], keys[:-1]))
    new_mask = keys & ~prev
    idx = np.flatnonzero(new_mask != 0)
    return idx, frames.times[idx].astype(np.int64), new_mask[idx]


def _classify(abs_err: float, dist: float, radius: float, wg: float, wo: float, wm: float) -> str:
    if dist > radius:
        return JMISS
    if abs_err <= wg:
        return J300
    if abs_err <= wo:
        return J100
    if abs_err <= wm:
        return J50
    return JMISS


# ---------------------------------------------------------------------------
def _tracking_nested(obj: HitObject, events_real: list[tuple[float, str]],
                     head_action: str | None, press_frame: int,
                     frames: ReplayFrames, radius: float,
                     ot_real: float, dur_real: float, cpos):
    """(hits, total) for ticks+repeats+tail via the lazer tracking model."""
    # events_real: list of (time_real, kind) with kind in {"tick","repeat","tail"}
    if not events_real:
        return 0, 0
    if head_action is None:
        return 0, len(events_real)

    ft, fx, fy, fk = frames.times, frames.x, frames.y, frames.keys
    end_time = ot_real + dur_real

    def progress(t: float) -> float:
        return 0.0 if dur_real <= 0 else min(1.0, max(0.0, (t - ot_real) / dur_real))

    judged: dict[float, bool] = {}

    m = (float(fx[press_frame]), float(fy[press_frame]))
    press_t = float(ft[press_frame])
    in_expanded = _dist(m, cpos(progress(press_t))) <= radius * FOLLOW_AREA
    passed = [e for e in events_real if e[0] <= press_t]
    if in_expanded:
        all_in = all(_dist(m, cpos(progress(e[0]))) <= radius * FOLLOW_AREA for e in passed)
        for e in passed:
            judged[e[0]] = all_in
        tracking = all_in or (_dist(m, cpos(progress(press_t))) <= radius)
    else:
        for e in passed:
            judged[e[0]] = False
        tracking = False

    ticks_repeats = [e[0] for e in events_real if e[1] != "tail"]
    time_to_any: float | None = None
    last_pressed: set[str] = set()

    i1 = int(np.searchsorted(ft, end_time + 1.0, side="right"))
    for i in range(press_frame, i1):
        t = float(ft[i])
        if time_to_any is None:
            other = "R" if head_action == "L" else "L"
            if other not in last_pressed:
                time_to_any = t
        cur = _actions(int(fk[i]))
        last_pressed = cur
        valid = any(a == head_action for a in cur) if (time_to_any is None or t <= time_to_any) else True

        fr = radius * (FOLLOW_AREA if tracking else 1.0)
        in_pos = _dist((float(fx[i]), float(fy[i])), cpos(progress(t))) <= fr
        tracking = in_pos and valid

        for e_t, kind in events_real:
            if e_t in judged:
                continue
            trig = e_t - 36.0 if kind == "tail" else e_t
            if trig > t:
                continue
            if kind == "tail" and any(tr not in judged for tr in ticks_repeats):
                continue
            judged[e_t] = tracking

    hits = sum(1 for v in judged.values() if v)
    return hits, len(events_real)


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


# ---------------------------------------------------------------------------
FIRST_LAZER_VERSION = 30000000


def _is_legacy_version(frames: ReplayFrames) -> bool:
    """True for stable (pre-lazer) replays, whose data is stored in beatmap time."""
    return int(getattr(frames, "game_version", 0) or 0) < FIRST_LAZER_VERSION


def judge_replay(bm: Beatmap, frames: ReplayFrames, mods=(),
                 *, replay_map_time: bool | None = None) -> JudgeReport:
    """Judge a replay.

    ``replay_map_time`` selects the replay timestamp base. Stable replays store
    the cursor/key stream in *beatmap* time (unaffected by DT/HT), whereas the
    lazer client stores real (rate-scaled) time. When ``None`` it is inferred
    from the replay's game version. In map-time mode object/event times are used
    verbatim and the OD windows (defined in real time) are widened by the rate.
    """
    if replay_map_time is None:
        replay_map_time = _is_legacy_version(frames)
    rate = clock_rate(mods)
    flip = is_hr(mods)
    cs, ar, od, hp = effective_difficulty(bm.cs, bm.ar, bm.od, bm.hp, mods)
    radius = object_radius(cs)
    tscale = 1.0 if replay_map_time else 1.0 / rate
    wscale = rate if replay_map_time else 1.0
    wg, wo, wm, _ = (v * wscale for v in osu_hit_windows(od))

    def fy(y: float) -> float:
        return PLAYFIELD_H - y if flip else y

    def cpos(obj: HitObject, prog: float) -> tuple[float, float]:
        p = curve_position(obj, prog)
        return (p.x, fy(p.y))

    press_idx, press_times, press_mask = press_events(frames)
    n_press = len(press_times)
    used = np.zeros(n_press, dtype=bool)

    report = JudgeReport()

    for obj in bm.hit_objects:
        if obj.is_spinner:
            report.results.append(_judge_spinner(obj, frames, tscale, od))
            continue
        if not obj.is_clickable:
            continue

        ot = obj.time * tscale
        ox, oy = obj.x, fy(obj.y)
        lo = int(np.searchsorted(press_times, ot - wm))
        hi = int(np.searchsorted(press_times, ot + wm))
        best = None
        for pi in range(lo, hi):
            if used[pi]:
                continue
            pt = int(press_times[pi])
            cx, cy = cursor_at(frames, pt)
            dist = float(np.hypot(cx - ox, cy - oy))
            err = pt - ot
            kind = _classify(abs(err), dist, radius, wg, wo, wm)
            if kind == JMISS:
                continue
            # osu! consumes the *earliest* qualifying press, not the closest.
            best = (pi, kind, float(err), dist, pt)
            break

        if best is None:
            res = JudgementResult(obj.time, JMISS, None, None, None, is_slider=obj.is_slider)
            if obj.is_slider:
                total = len(_slider_events_real(bm, obj, tscale))
                res.tick_total = total
                res.tick_misses = total
            report.results.append(res)
            continue

        pi, kind, err, dist, pt = best
        used[pi] = True

        if not obj.is_slider:
            report.results.append(JudgementResult(obj.time, kind, err, dist, pt))
            continue

        haction = _head_action(int(press_mask[pi]))
        dur_real = (obj.span_duration * obj.span_count) * tscale
        events_real = _slider_events_real(bm, obj, tscale)
        hits, total = _tracking_nested(obj, events_real, haction, pi, frames,
                                       radius, ot, dur_real, lambda p: cpos(obj, p))
        report.results.append(JudgementResult(
            obj.time, kind, err, dist, pt,
            is_slider=True, dropped=(total - hits) > 0,
            tick_hits=hits, tick_total=total, tick_misses=total - hits,
        ))

    return report


def _slider_events_real(bm: Beatmap, obj: HitObject, tscale: float) -> list[tuple[float, str]]:
    out: list[tuple[float, str]] = []
    for e in slider_events(bm, obj):
        if e.type == SE.HEAD or e.type == SE.LEGACY_LAST_TICK:
            continue
        if e.type == SE.TICK:
            kind = "tick"
        elif e.type == SE.REPEAT:
            kind = "repeat"
        elif e.type == SE.TAIL:
            kind = "tail"
        else:
            continue
        out.append((e.time * tscale, kind))
    out.sort(key=lambda x: x[0])
    return out


# ---------------------------------------------------------------------------
def _judge_spinner(obj: HitObject, frames: ReplayFrames, tscale: float, od: float) -> JudgementResult:
    start = obj.time * tscale
    end = obj.end_time * tscale
    spins = _spinner_rotation(frames, start, end)
    required = _spins_required(od, end - start)
    progress = 1.0 if required <= 0 else spins / required
    if progress >= 1.0:
        kind = J300
    elif progress > 0.9:
        kind = J100
    elif progress > 0.75:
        kind = J50
    else:
        kind = JMISS
    return JudgementResult(obj.time, kind, None, None, None, is_spinner=True)


def _spinner_rotation(frames: ReplayFrames, start: float, end: float) -> float:
    lo = int(np.searchsorted(frames.times, start, side="left"))
    hi = int(np.searchsorted(frames.times, end, side="right"))
    if hi - lo < 2:
        return 0.0
    xs = frames.x[lo:hi].astype(np.float64)
    ys = frames.y[lo:hi].astype(np.float64)
    cx, cy = 256.0, 192.0
    angles = np.arctan2(ys - cy, xs - cx)
    d = np.diff(angles)
    d = np.where(d > np.pi, d - 2 * np.pi, d)
    d = np.where(d < -np.pi, d + 2 * np.pi, d)
    return abs(float(np.sum(d))) / (2 * np.pi)


def _spins_required(od: float, duration: float) -> int:
    # osu.Game.Rulesets.Osu/Objects/Spinner.cs: CLEAR_RPM_RANGE(90, 150, 225) at OD [0,5,10].
    rpm = difficulty_range(od, (90.0, 150.0, 225.0))
    return int(rpm / 60.0 * (duration / 1000.0) + 0.0001)
