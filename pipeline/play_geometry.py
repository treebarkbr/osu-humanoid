"""Slider/spinner *active-object* geometry + per-frame "act" features.

The base shard features (``pipeline/features.py``) only carry object *heads*
(``ot, ox, oy, oend, otype``).  During a slider the cursor is supposed to follow
the moving slider *ball*, and during a spinner it should orbit the spinner
centre -- neither of which the head-only features can express.  Without them the
aim net has nothing to follow (sliders get "abandoned") and nothing to orbit
(spinners "paralyse").

This module supplies that missing information in a compact, **learned** way
(no hand-coded motion): every frame gets an 8-vector describing the currently
active object relative to the cursor.

    act = [ dx_now, dy_now, dx_ahead, dy_ahead, kind, progress, spin_sin, spin_cos ]

  * ``dx_now/dy_now``     : vector from cursor to the active target *now*
                            (slider ball / spinner centre / next head), /POS
  * ``dx_ahead/dy_ahead`` : same, at ``t + LOOKAHEAD*DT`` -> gives the net the
                            ball's *velocity direction* so it can anticipate a
                            slider reversal instead of lagging behind the ball
  * ``kind``              : 0 = next head, 1 = slider ball, 2 = spinner  (->/2)
  * ``progress``          : fraction elapsed through the active object (0..1)
  * ``spin_sin/spin_cos`` : sin/cos of the cursor's angle around the spinner
                            centre (0 for non-spinners) -> lets the net learn to
                            keep the cursor circling

Slider paths are a property of the *beatmap*, not the replay, so instead of
re-extracting the 2543 replay shards we cache one small geometry file per
beatmap (154 of them) and compute ``act`` on demand in both training
(``pipeline/torch_data.py``) and inference (``scripts/rollout.py``).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.beatmap import Beatmap, parse_beatmap  # noqa: E402

SAMP = 64          # samples along each slider path (arc-length parameterised)
POS = 512.0
DT = 16.0
LOOKAHEAD = 2      # frames ahead for the velocity-direction vector
ACT_DIM = 8

GEOM_KEYS = ("paths", "span_dur", "span_count")


def geom_from_beatmap(bm: Beatmap) -> dict:
    """Per-object slider geometry: sampled absolute paths + span timing."""
    objs = bm.hit_objects
    m = len(objs)
    paths = np.zeros((m, SAMP, 2), dtype=np.float32)
    span_dur = np.zeros(m, dtype=np.float32)
    span_count = np.ones(m, dtype=np.int16)
    for i, o in enumerate(objs):
        if o.is_slider and o.path is not None and o.path.distance > 1e-3:
            span_dur[i] = float(o.span_duration)
            span_count[i] = int(o.span_count)
            for s in range(SAMP):
                p = o.path.position_at(s / (SAMP - 1))
                paths[i, s, 0] = o.x + p.x
                paths[i, s, 1] = o.y + p.y
        else:
            paths[i, :, 0] = o.x
            paths[i, :, 1] = o.y
    return {"paths": paths, "span_dur": span_dur, "span_count": span_count}


def save_geometry(geom: dict, path: str | Path) -> None:
    np.savez_compressed(path, **geom)


def load_geometry(path: str | Path) -> dict:
    with np.load(path) as d:
        return {k: d[k] for k in GEOM_KEYS}


def build_all(beatmaps_dir: str | Path, out_dir: str | Path) -> int:
    beatmaps_dir = Path(beatmaps_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for mp in sorted(beatmaps_dir.glob("*.osu")):
        bm = parse_beatmap(mp)
        save_geometry(geom_from_beatmap(bm), out_dir / f"{mp.stem}.npz")
        n += 1
        if n % 25 == 0:
            print(f"[{n}] {mp.stem}", flush=True)
    print(f"done. {n} beatmaps -> {out_dir}", flush=True)
    return n


def ball_pos(geom: dict, oi: np.ndarray, t: np.ndarray, ot: np.ndarray) -> np.ndarray:
    """Absolute slider-ball position for object indices ``oi`` at times ``t``.

    ``oi``/``t`` are equal-length arrays; returns ``(N, 2)``.  Non-sliders return
    their head.  Multi-span ping-pong is handled via span-index parity (matches
    osu!'s repeat behaviour).
    """
    oi = np.asarray(oi, dtype=np.int64)
    t = np.asarray(t, dtype=np.float64)
    sd = geom["span_dur"][oi].astype(np.float64)
    sc = geom["span_count"][oi].astype(np.int64)
    ot_o = ot[oi].astype(np.float64)
    u = np.where(sd > 0, (t - ot_o) / np.maximum(sd, 1e-9), 0.0)
    span = np.floor(u)
    local = u - span
    span = np.clip(span, 0, np.maximum(sc - 1, 0))
    local = np.where((span.astype(np.int64) & 1) == 1, 1.0 - local, local)
    idxf = np.clip(local, 0.0, 1.0) * (SAMP - 1)
    i0 = np.clip(np.floor(idxf).astype(np.int64), 0, SAMP - 1)
    i1 = np.clip(i0 + 1, 0, SAMP - 1)
    w = (idxf - i0)[:, None]
    p0 = geom["paths"][oi, i0]
    p1 = geom["paths"][oi, i1]
    return (p0 * (1.0 - w) + p1 * w).astype(np.float32)


def act_features(t, cx, cy, ot, ox, oy, oend, otype, geom, dt: float = DT,
                 lookahead: int = LOOKAHEAD) -> np.ndarray:
    """Per-frame ``(N, 8)`` active-object feature vector (see module docstring)."""
    t = np.asarray(t, dtype=np.float64)
    cx = np.asarray(cx, dtype=np.float64)
    cy = np.asarray(cy, dtype=np.float64)
    ot = np.asarray(ot, dtype=np.float64)
    ox = np.asarray(ox, dtype=np.float64)
    oy = np.asarray(oy, dtype=np.float64)
    oend = np.asarray(oend, dtype=np.float64)
    otype = np.asarray(otype)
    n = len(t)
    m = len(ot)
    nidx = np.searchsorted(ot, t, side="left")   # first object with ot >= t
    a = nidx - 1
    a_valid = (a >= 0) & (a < m)
    av = np.where(a_valid, a, 0)
    active = a_valid & (oend[av] >= t) & (otype[av] != 0)
    nidx_c = np.clip(nidx, 0, m - 1)

    tx = ox[nidx_c].astype(np.float64).copy()
    ty = oy[nidx_c].astype(np.float64).copy()
    tax = tx.copy()
    tay = ty.copy()

    sl = active & (otype[av] == 1)
    if np.any(sl):
        bp = ball_pos(geom, av, t, ot)
        bp2 = ball_pos(geom, av, t + lookahead * dt, ot)
        tx[sl] = bp[sl, 0]
        ty[sl] = bp[sl, 1]
        tax[sl] = bp2[sl, 0]
        tay[sl] = bp2[sl, 1]

    sp = active & (otype[av] == 2)
    tx[sp] = ox[av][sp]
    ty[sp] = oy[av][sp]
    tax[sp] = ox[av][sp]
    tay[sp] = oy[av][sp]

    kind = np.where(active, np.where(otype[av] == 1, 1.0, 2.0), 0.0)
    u = np.clip((t - ot[av]) / np.maximum(oend[av] - ot[av], 1e-9), 0.0, 1.0)
    u = np.where(active, u, 0.0)
    ang = np.arctan2(cy - oy[av], cx - ox[av])
    ssin = np.where(sp, np.sin(ang), 0.0)
    scos = np.where(sp, np.cos(ang), 0.0)

    act = np.stack([
        (tx - cx) / POS, (ty - cy) / POS,
        (tax - cx) / POS, (tay - cy) / POS,
        kind / 2.0, u, ssin, scos,
    ], axis=-1)
    return act.astype(np.float32)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--beatmaps", default="data/beatmaps")
    ap.add_argument("--out", default="data/geometry")
    a = ap.parse_args()
    build_all(a.beatmaps, a.out)
