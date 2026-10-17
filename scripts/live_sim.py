"""Real-time (live) play simulator.

``rollout.generate`` pre-simulates a whole map, then the viewer replays the
finished trajectory -- so switching maps freezes while the model rolls out.
``PlaySim`` instead advances one 16 ms frame at a time so a viewer can drive it
in wall-clock time. The two offline-only steps get causal replacements:

  * tap    -- offline picks the peak of ``tap_p`` inside a +/-150 ms window.
              Live we buffer 150 ms and decide the press at window close, then
              patch the (already recorded) key stream. Same result, no lookahead.
  * jitter -- offline centre-smooths the whole tremor array; live we use a
              trailing 3-tap average (causal, one-frame delay).

``LiveJudge`` scores objects incrementally as their windows close, reusing the
headless judge's per-object helpers so the counts match ``judge_replay``.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import torch

from pipeline import judge as J
from pipeline.beatmap import curve_position
from pipeline.hitwindows import object_radius, osu_hit_windows
from pipeline.humanize import sample_error
from pipeline.play_geometry import geom_from_beatmap
from pipeline.replay import ReplayFrames
from pipeline.torch_data import H, K, POS, TMS, W
import scripts.rollout as RO

DT = 16.0
LEFT = RO.LEFT
RIGHT = RO.RIGHT
HOLD_MS = RO.HOLD_MS
TAP_WINDOW = 150.0
LAZER_VERSION = RO.LAZER_VERSION


class PlaySim:
    def __init__(self, bm, aim, tap, *, od, ar, rng, params,
                 threshold=0.5, clip=60.0, momentum=False, vclip=60.0):
        self.bm = bm
        self.aim = aim
        self.tap = tap
        self.od = od
        self.ar = ar
        self.rng = rng
        self.params = params
        self.threshold = threshold
        self.clip = clip
        self.momentum = momentum
        self.vclip = vclip

        objs = [o for o in bm.hit_objects if o.is_clickable or o.is_spinner]
        self.objs = objs
        self.ot = np.array([o.time for o in objs], dtype=np.float64)
        self.oend = np.array([o.end_time for o in objs], dtype=np.float64)
        self.ox = np.array([o.x for o in objs], dtype=np.float64)
        self.oy = np.array([o.y for o in objs], dtype=np.float64)
        self.otype = np.array([1 if o.is_slider else (2 if o.is_spinner else 0)
                               for o in objs], dtype=np.float64)
        self.geom = geom_from_beatmap(bm)

        self.t0 = max(0.0, float(self.ot[0]) - 800.0)
        self.t1 = float(self.oend[-1]) + 400.0
        self.n = int((self.t1 - self.t0) // DT) + 1

        self.cx = float(self.ox[0])
        self.cy = float(self.oy[0])
        self.x0, self.y0 = self.cx, self.cy
        self.hist = np.zeros((W, 5), dtype=np.float32)
        self.hist[:, 0] = self.cx / POS
        self.hist[:, 1] = self.cy / POS
        self.vx = 0.0
        self.vy = 0.0
        self.oi = 0

        self.times = self.t0 + DT * np.arange(self.n)
        self.xs = np.zeros(self.n)
        self.ys = np.zeros(self.n)
        self.tag = np.zeros(self.n)
        self.keys = np.zeros(self.n, dtype=np.uint8)

        self.i = 0
        self.tap_next = 0
        self.presses = []          # (t_ms, release_ms, key_bits)
        self._frames = None
        self.alt = 0
        self._jz = deque([(0.0, 0.0)] * 3, maxlen=3)
        self.record = False
        self.states = []           # (hist, ctx, meta, act) per stepped frame

    # -- cursor sampling -------------------------------------------------
    def now(self) -> float:
        return float(self.times[self.i - 1]) if self.i > 0 else self.t0

    def done(self) -> bool:
        return self.i >= self.n

    def sample(self, t: float):
        if self.i <= 0:
            return self.x0, self.y0, False, 0
        idx = int((t - self.t0) // DT)
        idx = max(0, min(idx, self.i - 1))
        return (float(self.xs[idx]), float(self.ys[idx]),
                bool(self.keys[idx] & 0b1111), idx)

    def key_at(self, i: int) -> int:
        t = float(self.times[i])
        k = 0
        for tp, rel, kv in self.presses:
            if tp <= t:
                k = kv if t < rel else 0
            else:
                break
        return k

    # -- one frame -------------------------------------------------------
    def step(self) -> bool:
        if self.i >= self.n:
            return False
        i = self.i
        t = float(self.times[i])

        while self.oi < len(self.ot) and self.ot[self.oi] < t:
            self.oi += 1
        prev = self.oi - 1
        held = 1.0 if (prev >= 0 and self.otype[prev] != 0.0
                       and t <= self.oend[prev]) else 0.0

        ctx = RO._ctx(self.ox, self.oy, self.ot, self.otype, self.oi,
                      self.cx, self.cy, t)
        act = RO._act1(t, self.cx, self.cy, self.ot, self.ox, self.oy,
                       self.oend, self.otype, self.geom)
        dt_next = (self.ot[self.oi] - t) if self.oi < len(self.ot) else 1e6
        meta = np.array([self.od / 10.0, self.ar / 10.0, held,
                         float(np.clip(dt_next / TMS, 0.0, 1.0))], dtype=np.float32)
        if self.record:
            self.states.append((self.hist.copy(), ctx.copy(), meta.copy(),
                                np.asarray(act, dtype=np.float32).copy()))
        with torch.no_grad():
            d = self.aim(torch.from_numpy(self.hist[None]),
                         torch.from_numpy(ctx[None]),
                         torch.from_numpy(meta[None]),
                         torch.from_numpy(act[None])).numpy()[0]
            tp = float(torch.sigmoid(self.tap(
                torch.from_numpy(self.hist[None]),
                torch.from_numpy(ctx[None]),
                torch.from_numpy(meta[None]),
                torch.from_numpy(act[None]))).item())
        self.tag[i] = tp

        if self.momentum:
            a = np.clip(d, -self.clip, self.clip)
            self.vx = float(np.clip(self.vx + float(a[0]), -self.vclip, self.vclip))
            self.vy = float(np.clip(self.vy + float(a[1]), -self.vclip, self.vclip))
            dx_, dy_ = self.vx, self.vy
        else:
            d = np.clip(d, -self.clip, self.clip)
            dx_, dy_ = float(d[0]), float(d[1])

        ncx = float(np.clip(self.cx + dx_, 0.0, 512.0))
        ncy = float(np.clip(self.cy + dy_, 0.0, 384.0))
        dx_, dy_ = ncx - self.cx, ncy - self.cy
        self.cx, self.cy = ncx, ncy
        if self.momentum:
            self.vx, self.vy = dx_, dy_

        speed = float(np.hypot(dx_, dy_) / DT)
        amp = self.params.jitter_px * (0.5 + 0.5 * min(max(speed / 3.0, 0.0), 2.0))
        nz = self.rng.normal(0.0, 1.0, size=2)
        self._jz.append((nz[0] * amp, nz[1] * amp))
        jx = sum(v[0] for v in self._jz) / 3.0
        jy = sum(v[1] for v in self._jz) / 3.0
        self.xs[i], self.ys[i] = ncx + jx, ncy + jy

        self.hist = np.roll(self.hist, -1, axis=0)
        self.hist[-1] = (self.cx / POS, self.cy / POS,
                         (dx_ / DT) / POS, (dy_ / DT) / POS, held)

        self._decide_taps(i)
        self.keys[i] = self.key_at(i)
        self.i += 1
        return True

    def _decide_taps(self, i: int) -> None:
        while self.tap_next < len(self.ot):
            ot = float(self.ot[self.tap_next])
            if self.times[i] < ot + TAP_WINDOW:
                break
            lo = int(np.searchsorted(self.times, ot - TAP_WINDOW, side="left"))
            hi = min(int(np.searchsorted(self.times, ot + TAP_WINDOW, side="right")),
                     i + 1)
            if hi > lo:
                seg = self.tag[lo:hi]
                jj = lo + int(np.argmax(seg))
                if self.tag[jj] >= self.threshold:
                    t_press = (float(self.times[jj])
                               + sample_error(self.rng, self.params, 0.0))
                    cell = int(np.clip(round((t_press - self.t0) / DT), 0, self.n - 1))
                    kv = LEFT if self.alt == 0 else RIGHT
                    self.alt ^= 1
                    rel = float(self.times[cell]) + HOLD_MS
                    k = int(np.searchsorted(self.ot, self.times[cell] + 1.0)) - 1
                    if 0 <= k < len(self.ot) and abs(self.ot[k] - self.times[cell]) <= 90.0 \
                            and self.otype[k] != 0.0:
                        rel = max(rel, float(self.oend[k]) + 40.0)
                    self.presses.append((float(self.times[cell]), rel, kv))
                    rel_i = int(np.clip(round((rel - self.t0) / DT), cell, self.n - 1))
                    self.keys[cell:min(i + 1, rel_i)] = kv
            self.tap_next += 1

    def advance_to(self, t_now: float, max_steps: int = 512) -> int:
        steps = 0
        while self.i < self.n and self.times[self.i] <= t_now and steps < max_steps:
            self.step()
            steps += 1
        return steps

    # -- finished trajectory --------------------------------------------
    def finalize(self, upto: int | None = None) -> dict:
        k = self.i if upto is None else int(upto)
        return {
            "grid": self.times[:k].copy(),
            "x": self.xs[:k].copy(),
            "y": self.ys[:k].copy(),
            "keys": self.keys[:k].copy(),
            "objs": self.objs,
            "ot": self.ot, "ox": self.ox, "oy": self.oy,
            "oend": self.oend, "otype": self.otype, "geom": self.geom,
            "od": self.od, "ar": self.ar, "cs": self.bm.cs,
        }

    def frames(self) -> ReplayFrames:
        # Built once, referencing the live arrays (no per-call copy). Safe
        # because the judge only reads frames up to an object's resolve time,
        # which are already written by then.
        if self._frames is None:
            self._frames = ReplayFrames(
                times=self.times, x=self.xs, y=self.ys, keys=self.keys,
                mods=(), count300=0, count100=0, count50=0, count_miss=0,
                max_combo=0, beatmap_hash="", username="osu-humanoid",
                game_version=LAZER_VERSION,
            )
        return self._frames


class LiveJudge:
    """Incremental judge (nomod) -- resolves objects once their window closes."""

    def __init__(self, bm):
        self.bm = bm
        self.objs = bm.hit_objects
        self.radius = object_radius(bm.cs)
        self.wg, self.wo, self.wm, _ = osu_hit_windows(bm.od)
        self.results = []
        self.used = set()
        self.obj_i = 0

    def _cpos(self, obj, prog):
        p = curve_position(obj, prog)
        return (p.x, p.y)

    def update(self, sim: PlaySim) -> None:
        if sim.i <= 0 or self.obj_i >= len(self.objs):
            return
        t = float(sim.times[sim.i - 1])
        frames = press_times = press_mask = None
        while self.obj_i < len(self.objs):
            o = self.objs[self.obj_i]
            if o.is_spinner:
                resolve = o.end_time + 10.0
            elif o.is_slider:
                resolve = o.end_time + 60.0
            else:
                resolve = o.time + TAP_WINDOW + self.wm + 10.0
            if t < resolve:
                break
            if frames is None:               # lazy: only when something resolves
                frames = sim.frames()
                _, press_times, press_mask = J.press_events(frames)
            self.results.append(self._one(o, frames, press_times, press_mask))
            self.obj_i += 1

    def _one(self, o, frames, press_times, press_mask):
        if o.is_spinner:
            return J._judge_spinner(o, frames, 1.0, self.bm.od)
        if not o.is_clickable:
            return None
        ot = float(o.time)
        radius = self.radius
        lo = int(np.searchsorted(press_times, ot - self.wm))
        hi = int(np.searchsorted(press_times, ot + self.wm))
        best = None
        for pi in range(lo, hi):
            if pi in self.used:
                continue
            pt = int(press_times[pi])
            cx, cy = J.cursor_at(frames, pt)
            dist = float(np.hypot(cx - o.x, cy - o.y))
            err = pt - ot
            kind = J._classify(abs(err), dist, radius, self.wg, self.wo, self.wm)
            if kind == J.JMISS:
                continue
            best = (pi, kind, float(err), dist, pt)
            break
        if best is None:
            r = J.JudgementResult(o.time, J.JMISS, None, None, None, is_slider=o.is_slider)
            if o.is_slider:
                total = len(J._slider_events_real(self.bm, o, 1.0))
                r.tick_total = total
                r.tick_misses = total
            return r
        pi, kind, err, dist, pt = best
        self.used.add(pi)
        if not o.is_slider:
            return J.JudgementResult(o.time, kind, err, dist, pt)
        haction = J._head_action(int(press_mask[pi]))
        dur = o.span_duration * o.span_count
        events = J._slider_events_real(self.bm, o, 1.0)
        hits, total = J._tracking_nested(o, events, haction, pi, frames, radius,
                                         ot, dur, lambda p: self._cpos(o, p))
        return J.JudgementResult(
            o.time, kind, err, dist, pt,
            is_slider=True, dropped=(total - hits) > 0,
            tick_hits=hits, tick_total=total, tick_misses=total - hits)
