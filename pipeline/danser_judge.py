"""Faithful port of danser-go's osu!standard **stable** judgement.

Replaces the lazer-modelled judge in ``pipeline/judge.py``.  Direct
translation of the following danser-go sources:

  app/rulesets/osu/ruleset.go     (frame loop, CanBeHit, GetResultForDelta)
  app/rulesets/osu/circle.go      (single-press circles + post miss)
  app/rulesets/osu/slider.go      (stable head/tick/repeat/end + rate result)
  app/rulesets/osu/spinner.go     (stable velocity-integration spin model)
  app/beatmap/stackleniency.go    (OsuBeatmapProcessor stacking)
  app/beatmap/difficulty/difficulty.go (OD/AR/CS derived values)

The headline fix (what the user reported): a slider's accuracy result is
derived from the fraction of score points actually collected::

    rate = scored / (len(points) + 1)
    300 if rate == 1 else 100 if rate >= 0.5 else 50 if rate > 0 else miss

Previously a slider was scored by its *head* error, so abandoning a slider
half-way still counted as a hit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .beatmap import Beatmap, HitObject, curve_position, slider_events
from . import slider_events as SE
from .hitwindows import object_radius
from .mods import clock_rate, effective_difficulty, is_hr
from .replay import ReplayFrames

FRAME_TIME = 1000.0 / 60.0
TOLERANCE_2B = 3
STACK_DISTANCE = 3.0

B_LEFT, B_RIGHT = 1, 2
LEFT_BITS = 0b0101   # M1 | K1
RIGHT_BITS = 0b1010  # M2 | K2

K300, K100, K50, KMISS = "300", "100", "50", "miss"

P_TICK, P_REPEAT, P_END = 0, 1, 2
CLICK, SHAKE, IGNORED = 0, 1, 2


def _dr(diff: float, lo: float, mid: float, hi: float) -> float:
    """difficulty.DifficultyRate (float32-cast like the Go original)."""
    d = float(np.float32(diff))
    if d > 5.0:
        return mid + (hi - mid) * (d - 5.0) / 5.0
    if d < 5.0:
        return mid - (mid - lo) * (5.0 - d) / 5.0
    return mid


@dataclass
class _Diff:
    radius: float
    hit300: float
    hit100: float
    hit50: float
    spinner_ratio: float
    preempt: float
    speed: float      # effective speed for spinner GetModifiedTime
    tscale: float     # object time (map ms) -> frame time-base
    wscale: float     # window scale into frame time-base


def _make_diff(cs, ar, od, mods, replay_map_time: bool) -> _Diff:
    rate = clock_rate(mods)
    w300 = int(_dr(od, 80, 50, 20))
    w100 = int(_dr(od, 140, 100, 60))
    w50 = int(_dr(od, 200, 150, 100))
    if replay_map_time:
        # Stable replays store frames in beatmap (unmodified) time and the
        # hit windows are compared directly in that same time base -- DT/HT do
        # NOT widen or narrow them (verified against 2000+ official replays).
        tscale, wscale, speed = 1.0, 1.0, 1.0
    else:
        tscale, wscale, speed = 1.0 / rate, 1.0, 1.0
    return _Diff(
        radius=object_radius(cs),
        hit300=w300 * wscale,
        hit100=w100 * wscale,
        hit50=w50 * wscale,
        spinner_ratio=_dr(od, 3, 5, 7.5),
        preempt=math.floor(_dr(ar, 1800, 1200, 450)),
        speed=speed,
        tscale=tscale,
        wscale=wscale,
    )


# ---------------------------------------------------------------------------
# stacking (OsuBeatmapProcessor, verbatim structure from danser)
# ---------------------------------------------------------------------------
def _stacked(x, y, hr, stack_off):
    if hr:
        y = 384.0 - y
    return x - stack_off, y - stack_off


def _compute_stacking(objs, version: int, preempt: float, leniency: float) -> list[int]:
    n = len(objs)
    si = [0] * n
    if n == 0:
        return si
    thr = int(math.floor(preempt * leniency))

    starts = [o.time for o in objs]
    ends = [o.end_time for o in objs]
    spx = [o.x for o in objs]
    spy = [o.y for o in objs]
    epx = [0.0] * n
    epy = [0.0] * n
    is_slider = [o.is_slider for o in objs]
    is_spinner = [o.is_spinner for o in objs]
    for i, o in enumerate(objs):
        if o.is_slider and o.path is not None:
            if o.span_count % 2 == 1:
                p = curve_position(o, 1.0)
                epx[i], epy[i] = p.x, p.y
            else:
                epx[i], epy[i] = o.x, o.y

    def dst(x1, y1, x2, y2):
        return math.hypot(x1 - x2, y1 - y2)

    if version >= 6:
        # --- first pass ---
        extended_end = n - 1
        i = n - 1
        while i > 0:
            base = i
            k = base + 1
            while k < n:
                if is_spinner[base]:
                    break
                if is_spinner[k]:
                    k += 1
                    continue
                if starts[k] - ends[base] > thr:
                    break
                if (dst(spx[base], spy[base], spx[k], spy[k]) < STACK_DISTANCE
                        or (is_slider[base] and dst(epx[base], epy[base], spx[k], spy[k]) < STACK_DISTANCE)):
                    base = k
                    si[k] = 0
                k += 1
            if base > extended_end:
                extended_end = base
                if extended_end == n - 1:
                    break
            i -= 1

        # --- second pass (objectN walks backwards, becomes objectI) ---
        extended_start = 0
        i = extended_end
        while i > 0:
            if si[i] != 0 or is_spinner[i]:
                i -= 1
                continue
            if not is_slider[i]:
                # circles
                ref = i
                k = i - 1
                while k >= 0:
                    if is_spinner[k]:
                        k -= 1
                        continue
                    if starts[ref] - ends[k] > thr:
                        break
                    if k < extended_start:
                        si[k] = 0
                        extended_start = k
                    if is_slider[k] and dst(epx[k], epy[k], spx[ref], spy[ref]) < STACK_DISTANCE:
                        offset = si[ref] - si[k] + 1
                        for j in range(k + 1, ref + 1):
                            if dst(epx[k], epy[k], spx[j], spy[j]) < STACK_DISTANCE:
                                si[j] -= offset
                        break
                    if dst(spx[k], spy[k], spx[ref], spy[ref]) < STACK_DISTANCE:
                        si[k] = si[ref] + 1
                        ref = k
                    k -= 1
            else:
                # sliders
                ref = i
                k = i - 1
                while k >= 0:
                    if is_spinner[k]:
                        k -= 1
                        continue
                    if starts[ref] - starts[k] > thr:
                        break
                    if dst(epx[k], epy[k], spx[ref], spy[ref]) < STACK_DISTANCE:
                        si[k] = si[ref] + 1
                        ref = k
                    k -= 1
            i -= 1
    else:
        # --- applyOldStacking ---
        for i in range(n):
            if si[i] == 0 or is_slider[i]:
                slider_stack = 0
                start_time = ends[i]
                for k in range(i + 1, n):
                    if starts[k] - thr > start_time:
                        break
                    if dst(spx[k], spy[k], spx[i], spy[i]) < STACK_DISTANCE:
                        si[i] += 1
                        start_time = ends[k]
                    elif dst(spx[k], spy[k], epx[i], epy[i]) < STACK_DISTANCE:
                        slider_stack += 1
                        si[k] -= slider_stack
                        start_time = ends[k]

    for i in range(n):
        if is_spinner[i]:
            si[i] = 0
    return si


# ---------------------------------------------------------------------------
def _ball_path_progress(u: float, span_count: int) -> float:
    if span_count <= 1:
        return min(1.0, max(0.0, u))
    sp = span_count
    idx = min(int(u * sp), sp - 1)
    local = u * sp - idx
    return local if idx % 2 == 0 else 1.0 - local


# ---------------------------------------------------------------------------
class _Base:
    is_slider = False
    is_spinner = False
    stack_off = 0.0

    def click(self, t, fr):
        pass

    def update_for(self, t, fr):
        pass

    def post_for(self, t, fr):
        pass

    def is_hit_flag(self) -> bool:
        return True

    def done(self) -> bool:
        return True


class _Circle(_Base):
    def __init__(self, obj, idx, diff, hr, stack_off):
        self.obj = obj
        self.idx = idx
        self.start = obj.time * diff.tscale
        self.end = obj.end_time * diff.tscale
        self.sx, self.sy = _stacked(obj.x, obj.y, hr, stack_off)
        self.stack_off = stack_off
        self.diff = diff
        self.is_hit = False
        self.kind = None
        self.hit_error = None
        self.dist = None
        self.press_time = None

    def click(self, t, fr):
        if self.is_hit:
            return
        if not (fr.left_e or fr.right_e):
            return
        action = fr.can_be_hit(t, self)
        in_range = math.hypot(fr.x - self.sx, fr.y - self.sy) <= self.diff.radius
        if in_range:
            if action == CLICK:
                if fr.left_e:
                    fr.left_e = False
                elif fr.right_e:
                    fr.right_e = False
                self.kind, _ = fr.result_for_delta(abs(t - self.end))
                self.hit_error = t - self.start
                self.dist = math.hypot(fr.x - self.sx, fr.y - self.sy)
                self.press_time = int(t)
                self.is_hit = True
            else:
                fr.left_e = False
                fr.right_e = False

    def post_for(self, t, fr):
        if t > self.end + self.diff.hit50 and not self.is_hit:
            self.kind = KMISS
            self.is_hit = True

    def is_hit_flag(self):
        return self.is_hit

    def done(self):
        return self.is_hit


class _Slider(_Base):
    is_slider = True

    def __init__(self, obj, idx, diff, hr, stack_off, points):
        self.obj = obj
        self.idx = idx
        self.diff = diff
        self.start = obj.time * diff.tscale
        self.dur = obj.span_count * obj.span_duration * diff.tscale
        self.end = self.start + self.dur
        self.sx, self.sy = _stacked(obj.x, obj.y, hr, stack_off)
        self.hr = hr
        self.stack_off = stack_off
        self.span_count = obj.span_count
        self.points = points            # [(time_T, kind), ...] ascending
        self.npoints = len(points)
        self.step = max(1, self.npoints // max(1, obj.span_count))

        self.is_hit = False
        self.is_start_hit = False
        self.start_result = KMISS
        self.sliding = False
        self.slide_start = 0.0
        self.down_button = 0
        self.scored = 0
        self.missed = 0
        self.passed = 0
        self.kind = None
        self.hit_error = None
        self.dist = None
        self.press_time = None

    def ball(self, t):
        u = 0.0 if self.dur <= 0 else (t - self.start) / self.dur
        u = min(1.0, max(0.0, u))
        p = curve_position(self.obj, _ball_path_progress(u, self.span_count))
        return _stacked(p.x, p.y, self.hr, self.stack_off)

    def click(self, t, fr):
        if self.is_start_hit:
            return
        if not (fr.left_e or fr.right_e):
            return
        action = fr.can_be_hit(t, self)
        in_range = math.hypot(fr.x - self.sx, fr.y - self.sy) <= self.diff.radius
        if in_range:
            if action == CLICK:
                if fr.left_e:
                    fr.left_e = False
                elif fr.right_e:
                    fr.right_e = False
                if fr.left_held:
                    self.down_button = B_LEFT
                elif fr.right_held:
                    self.down_button = B_RIGHT
                else:
                    self.down_button = fr.mouse_down_button
                self.start_result, _ = fr.result_for_delta(abs(t - self.start))
                self.is_start_hit = True
                self.press_time = int(t)
                self.hit_error = t - self.start
                self.dist = math.hypot(fr.x - self.sx, fr.y - self.sy)
            else:
                fr.left_e = False
                fr.right_e = False

    def update_for(self, t, fr):
        if self.is_hit or t < self.start:
            return
        mdb = fr.mouse_down_button
        swap = fr.game_down and not (
            fr.last_button == (B_LEFT | B_RIGHT) and fr.last_button2 == mdb)
        acceptable = False
        if fr.game_down:
            if self.down_button == 0 or (mdb != (B_LEFT | B_RIGHT) and swap):
                self.down_button = 0
                if fr.left_held:
                    self.down_button = B_LEFT
                elif fr.right_held:
                    self.down_button = B_RIGHT
                else:
                    self.down_button = mdb
                acceptable = True
            elif (mdb & self.down_button) > 0:
                acceptable = True
        else:
            self.down_button = 0
        acceptable = acceptable or swap

        bx, by = self.ball(t)
        rad = self.diff.radius * (2.4 if self.sliding else 1.0)
        allowable = acceptable and (fr.x - bx) ** 2 + (fr.y - by) ** 2 < rad * rad
        if allowable and not self.sliding:
            self.sliding = True
            self.slide_start = t

        self._process_ticks(t, fr)

        if not allowable and self.sliding and self.scored + self.missed < self.npoints:
            self.sliding = False

    def _process_ticks(self, t, fr):
        pts = self.points
        n = self.npoints
        p = self.passed
        while p < n and pts[p][0] <= t:
            p += 1
        self.passed = p
        if self.scored + self.missed < p:
            idx = self.scored + self.missed
            ptime = pts[idx][0]
            rad = self.diff.radius * 2.4
            ok = False
            if self.sliding and self.slide_start <= ptime and fr.held_at(ptime):
                cx, cy = fr.cursor_at(ptime)
                bx, by = self.ball(ptime)
                ok = (cx - bx) ** 2 + (cy - by) ** 2 < rad * rad
            if ok:
                self.scored += 1
            else:
                self.missed += 1

    def post_for(self, t, fr):
        if not self.is_start_hit and t > self.start + self.diff.hit50:
            self.is_start_hit = True
            self.start_result = KMISS
            if fr.left_held:
                self.down_button = B_LEFT
            elif fr.right_held:
                self.down_button = B_RIGHT
            else:
                self.down_button = fr.mouse_down_button
        if t >= self.end and not self.is_hit:
            if self.start_result != KMISS:
                self.scored += 1
            rate = self.scored / (self.npoints + 1)
            if rate == 1.0:
                self.kind = K300
            elif rate >= 0.5:
                self.kind = K100
            elif rate > 0.0:
                self.kind = K50
            else:
                self.kind = KMISS
            self.is_hit = True

    def is_hit_flag(self):
        return self.is_hit

    def done(self):
        return self.is_hit and self.is_start_hit


class _Spinner(_Base):
    is_spinner = True

    def __init__(self, obj, idx, diff, hr, stack_off):
        self.obj = obj
        self.idx = idx
        self.diff = diff
        self.start = obj.time * diff.tscale
        self.end = obj.end_time * diff.tscale
        self.dur = self.end - self.start
        self.stack_off = stack_off
        self.requirement = int(self.dur / 1000.0 * diff.spinner_ratio)
        self.max_accel = 0.00008 + max(0.0, (5000.0 - self.dur) / 1000.0 / 2000.0)

        self.cur_v = 0.0
        self.theo_v = 0.0
        self.rpm = 0.0
        self.rcount_f = 0.0
        self.rcount = 0
        self.last_rcount = 0
        self.scoring_rcount = 0
        self.last_angle = 0.0
        self.frame_variance = FRAME_TIME
        self.zero_count = 0
        self.updated_before = False
        self.finished = False
        self.kind = None
        self.last_frame_time = 0.0

    def _mod(self, dt):
        return dt / self.diff.speed if self.diff.speed else dt

    def update_for(self, t, fr):
        if self.finished:
            return
        if self.start < t < self.end:
            dt = t - self.last_frame_time
            if self.last_frame_time == 0:
                dt = FRAME_TIME
            max_accel_frame = self._mod(self.max_accel * dt)
            if self.theo_v > self.cur_v:
                self.cur_v += min(self.theo_v - self.cur_v, max_accel_frame)
            else:
                self.cur_v += max(self.theo_v - self.cur_v, -max_accel_frame)
            self.cur_v = max(-0.05, min(self.cur_v, 0.05))

            decay1 = math.pow(0.9, dt / FRAME_TIME)
            self.rpm = self.rpm * decay1 + (1.0 - decay1) * (abs(self.cur_v) * 1000.0) / (math.pi * 2) * 60.0

            mouse_angle = math.atan2(fr.y - self.cy, fr.x - self.cx)
            if not self.updated_before:
                self.last_angle = mouse_angle
                self.updated_before = True
            angle_diff = mouse_angle - self.last_angle
            if mouse_angle - self.last_angle < -math.pi:
                angle_diff = 2 * math.pi + mouse_angle - self.last_angle
            elif self.last_angle - mouse_angle < -math.pi:
                angle_diff = -2 * math.pi - self.last_angle + mouse_angle

            decay = math.pow(0.999, dt)
            self.frame_variance = decay * self.frame_variance + (1 - decay) * dt

            if angle_diff == 0:
                self.zero_count += 1
                if self.zero_count < 2:
                    self.theo_v /= 3
                else:
                    self.theo_v = 0.0
            else:
                self.zero_count = 0
                if not fr.game_down:
                    angle_diff = 0.0
                if abs(angle_diff) < math.pi:
                    if self._mod(self.frame_variance) > FRAME_TIME * 1.04:
                        self.theo_v = angle_diff / self._mod(dt) if dt > 0 else 0.0
                    else:
                        self.theo_v = angle_diff / FRAME_TIME
                else:
                    self.theo_v = 0.0
            self.last_angle = mouse_angle

            rot_add = self.cur_v * dt
            self.rcount_f += abs(rot_add) / math.pi
            self.rcount = int(self.rcount_f)
            if self.rcount != self.last_rcount:
                self.scoring_rcount += 1
                self.last_rcount = self.rcount
        self.last_frame_time = t

    def post_for(self, t, fr):
        if t >= self.end and not self.finished:
            req = self.requirement
            if req == 0 or self.scoring_rcount >= req + 1:
                self.kind = K300
            elif self.scoring_rcount >= req - 1:
                self.kind = K100
            elif self.scoring_rcount >= req // 4:
                self.kind = K50
            else:
                self.kind = KMISS
            self.finished = True

    def is_hit_flag(self):
        return self.finished

    def done(self):
        return self.finished

    # spinner needs a centre for the angle math
    cx = 256.0
    cy = 192.0


# ---------------------------------------------------------------------------
class _Frame:
    __slots__ = ("x", "y", "game_down", "mouse_down_button", "left_e", "right_e",
                 "left_held", "right_held", "last_button", "last_button2",
                 "is_replay", "diff", "processed", "cursor_at", "held_at")

    def __init__(self, diff):
        self.diff = diff
        self.is_replay = True
        self.processed: list[_Base] = []
        self.x = 0.0
        self.y = 0.0
        self.game_down = False
        self.mouse_down_button = 0
        self.left_e = False
        self.right_e = False
        self.left_held = False
        self.right_held = False
        self.last_button = 0
        self.last_button2 = 0
        self.cursor_at = None
        self.held_at = None

    def result_for_delta(self, delta):
        d = int(delta)
        if d < self.diff.hit300:
            return K300, 300
        if d < self.diff.hit100:
            return K100, 100
        if d < self.diff.hit50:
            return K50, 50
        return KMISS, 0

    def can_be_hit(self, t, obj):
        proc = self.processed
        index = -1
        for i, g in enumerate(proc):
            if g is obj:
                index = i
                break
        if isinstance(obj, _Circle) and index > 0:
            prev = proc[index - 1]
            if prev.stack_off > 0 and not prev.is_hit_flag():
                return IGNORED
        for i, g in enumerate(proc):
            if g is obj:
                break
            if not g.is_hit_flag() and g.end + TOLERANCE_2B < obj.start:
                return SHAKE
        if abs(t - obj.start) >= 400.0:
            return SHAKE
        return CLICK


# ---------------------------------------------------------------------------
def judge_danser(bm: Beatmap, frames: ReplayFrames, mods=(),
                 *, replay_map_time: bool | None = None):
    """Judge a replay with the danser stable ruleset -> JudgeReport."""
    from .judge import JudgeReport, JudgementResult, _is_legacy_version

    if replay_map_time is None:
        replay_map_time = _is_legacy_version(frames)

    flip = is_hr(mods)
    cs, ar, od, hp = effective_difficulty(bm.cs, bm.ar, bm.od, bm.hp, mods)
    diff = _make_diff(cs, ar, od, mods, replay_map_time)

    objs = list(bm.hit_objects)
    n = len(objs)
    if n == 0:
        return JudgeReport()

    preempt_map = math.floor(_dr(bm.ar, 1800, 1200, 450))
    si = _compute_stacking(objs, bm.format_version, preempt_map, bm.stack_leniency)
    scale = diff.radius / 10.0  # one stack level = r/10 in each axis

    states: list[_Base] = []
    for i, o in enumerate(objs):
        off = si[i] * scale
        if o.is_spinner:
            states.append(_Spinner(o, i, diff, flip, 0.0))
        elif o.is_slider:
            pts = []
            for e in slider_events(bm, o):
                if e.type == SE.TICK:
                    pts.append((e.time * diff.tscale, P_TICK))
                elif e.type == SE.REPEAT:
                    pts.append((e.time * diff.tscale, P_REPEAT))
                elif e.type == SE.LEGACY_LAST_TICK:
                    pts.append((e.time * diff.tscale, P_END))
            pts.sort(key=lambda p: p[0])
            states.append(_Slider(o, i, diff, flip, off, pts))
        elif o.is_circle:
            states.append(_Circle(o, i, diff, flip, off))

    times = frames.times
    xs = frames.x.astype(np.float64)
    ys = frames.y.astype(np.float64)
    keys = frames.keys.astype(np.int32)
    nf = len(times)
    lb_arr = (keys & LEFT_BITS) != 0
    rb_arr = (keys & RIGHT_BITS) != 0
    if nf:
        prev_lb = np.concatenate(([False], lb_arr[:-1]))
        prev_rb = np.concatenate(([False], rb_arr[:-1]))
    else:
        prev_lb = lb_arr
        prev_rb = rb_arr
    left_edge = lb_arr & ~prev_lb
    right_edge = rb_arr & ~prev_rb
    transition = (lb_arr != prev_lb) | (rb_arr != prev_rb)

    fr = _Frame(diff)
    fade = [s.start - diff.preempt * diff.wscale for s in states]
    q_idx = 0
    processed = fr.processed

    def cursor_at(t):
        if nf == 0:
            return 0.0, 0.0
        i = int(np.searchsorted(times, t))
        if i <= 0:
            return float(xs[0]), float(ys[0])
        if i >= nf:
            return float(xs[-1]), float(ys[-1])
        t0 = float(times[i - 1]); t1 = float(times[i])
        f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        return (float(xs[i - 1]) + (float(xs[i]) - float(xs[i - 1])) * f,
                float(ys[i - 1]) + (float(ys[i]) - float(ys[i - 1])) * f)

    def held_at(t):
        if nf == 0:
            return False
        i = int(np.searchsorted(times, t, side="right")) - 1
        if i < 0:
            i = 0
        k = int(keys[i])
        return bool(k & LEFT_BITS) or bool(k & RIGHT_BITS)

    fr.cursor_at = cursor_at
    fr.held_at = held_at

    for fi in range(nf):
        t = float(times[fi])
        if processed:
            processed[:] = [s for s in processed if not s.done()]
        while q_idx < n and fade[q_idx] <= t:
            processed.append(states[q_idx])
            q_idx += 1

        lb = bool(lb_arr[fi]); rb = bool(rb_arr[fi])
        fr.x = xs[fi]; fr.y = ys[fi]
        fr.left_e = bool(left_edge[fi])
        fr.right_e = bool(right_edge[fi])
        fr.left_held = bool(left_edge[fi])   # danser leftCond (new press)
        fr.right_held = bool(right_edge[fi])
        fr.game_down = lb or rb
        fr.mouse_down_button = (B_LEFT if lb else 0) | (B_RIGHT if rb else 0)
        if transition[fi]:
            fr.last_button2 = fr.last_button
            fr.last_button = fr.mouse_down_button

        for s in processed:
            s.click(t, fr)
        for s in processed:
            s.update_for(t, fr)
        for s in processed:
            s.post_for(t, fr)

    tail = float(times[-1]) + 5.0 * diff.hit50 + 10000.0 if nf else 1e12
    for s in states:
        if not s.done():
            s.post_for(tail, fr)

    report = JudgeReport()
    for s in states:
        o = s.obj
        if s.is_spinner:
            report.results.append(JudgementResult(
                o.time, s.kind or KMISS, None, None, None, is_spinner=True,
                spin_count=s.scoring_rcount, spin_req=s.requirement))
        elif s.is_slider:
            report.results.append(JudgementResult(
                o.time, s.kind or KMISS, s.hit_error, s.dist, s.press_time,
                is_slider=True,
                dropped=(s.missed > 0 or s.start_result == KMISS),
                tick_hits=s.scored, tick_total=s.npoints, tick_misses=s.missed))
        else:
            report.results.append(JudgementResult(
                o.time, s.kind or KMISS, s.hit_error, s.dist, s.press_time))
    return report
