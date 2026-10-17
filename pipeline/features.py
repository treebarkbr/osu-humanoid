"""Feature extraction: replay + beatmap -> uniform-grid numpy arrays.

Everything here is expressed in **map time** (the beatmap's own timeline). This
is the mod-invariant base: DT/HT only change the clock, so a player's cursor
*path* as a function of map time is the same as it would be nomod -- only the
real-time hit windows differ (handled by the judge). Working in map time lets
every replay, DT/HT included, train a single consistent motion model.

    stable replay : frames are already in map time
    lazer  replay : frames are in real time -> map = real * rate
    objects       : obj.time is already map time

HR is normalised to true (un-flipped) space: stable stores an HR cursor in the
flipped space, so we mirror the cursor back (384 - y) and leave objects as-is.
Every shard therefore lives in one consistent coordinate space -- the same one
``scripts/rollout.py`` generates in -- so no HR flag is needed in the features.

Per replay we emit one ``.npz`` with a fixed resample step:

  scalar meta : od, ar, cs, rate, dt, stable, game_version
  frame arrays (len N)   : t, x, y, vx, vy, held, press, next_idx, dt_next
  object arrays (len M)  : ot, ox, oy, oend, otype
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.beatmap import Beatmap, parse_beatmap  # noqa: E402
from pipeline.judge import _is_legacy_version, press_events  # noqa: E402
from pipeline.mods import clock_rate, effective_difficulty, parse_mods  # noqa: E402
from pipeline.replay import ReplayFrames, load_replay  # noqa: E402

PLAYFIELD_W, PLAYFIELD_H = 512.0, 384.0
DT_MS = 16.0  # resample step (~60 fps)


def _resample(times: np.ndarray, x: np.ndarray, y: np.ndarray,
              keys: np.ndarray, t0: float, t1: float, dt: float):
    """Cursor/keys onto a uniform grid: linear-interp position, held key state."""
    grid = np.arange(t0, t1 + dt, dt, dtype=np.float64)
    idx = np.searchsorted(times, grid, side="right") - 1
    idx = np.clip(idx, 0, len(times) - 1)
    nxt = np.clip(idx + 1, 0, len(times) - 1)

    t = times.astype(np.float64)
    t_a, t_b = t[idx], t[nxt]
    span = np.where(t_b > t_a, t_b - t_a, 1.0)
    w = np.clip((grid - t_a) / span, 0.0, 1.0)
    xr = x[idx] * (1.0 - w) + x[nxt] * w
    yr = y[idx] * (1.0 - w) + y[nxt] * w
    held = ((keys[idx] & 0b1111) > 0).astype(np.uint8)
    return grid.astype(np.float32), xr.astype(np.float32), yr.astype(np.float32), held


def extract(bm: Beatmap, frames: ReplayFrames, mods=()) -> dict:
    mods_bits = parse_mods(mods) if mods else parse_mods(frames.mods)
    rate = clock_rate(mods_bits)
    stable = _is_legacy_version(frames)
    flip = bool(mods_bits & 16)  # HR
    cs, ar, od, _hp = effective_difficulty(bm.cs, bm.ar, bm.od, bm.hp, mods_bits)

    # --- frames -> map time --------------------------------------------
    ft = frames.times.astype(np.float64)
    if not stable:
        ft = ft * rate  # lazer stores real time
    fx = frames.x.astype(np.float64)
    fy = frames.y.astype(np.float64)
    if flip:
        fy = PLAYFIELD_H - fy  # HR stores cursor in flipped space -> true space
    if len(ft) < 2:
        raise ValueError("replay too short")

    grid, gx, gy, gheld = _resample(ft, fx, fy, frames.keys, ft[0], ft[-1], DT_MS)
    gx = np.clip(gx, 0.0, PLAYFIELD_W)
    gy = np.clip(gy, 0.0, PLAYFIELD_H)

    dt = np.float32(DT_MS)
    gvx = np.gradient(gx, DT_MS).astype(np.float32)
    gvy = np.gradient(gy, DT_MS).astype(np.float32)

    # --- press edges as grid booleans ----------------------------------
    press = np.zeros(len(grid), dtype=np.uint8)
    _pi, press_times, _pm = press_events(frames)
    pt_map = press_times.astype(np.float64) * (1.0 if stable else rate)
    cell = np.searchsorted(grid, pt_map, side="right") - 1
    cell = cell[(cell >= 0) & (cell < len(grid))]
    press[cell] = 1

    # --- objects -> map time + HR flip ---------------------------------
    m = len(bm.hit_objects)
    ot = np.zeros(m, dtype=np.float32)
    ox = np.zeros(m, dtype=np.float32)
    oy = np.zeros(m, dtype=np.float32)
    oend = np.zeros(m, dtype=np.float32)
    otype = np.zeros(m, dtype=np.int8)
    for i, o in enumerate(bm.hit_objects):
        ot[i] = float(o.time)
        ox[i] = o.x
        oy[i] = o.y  # objects stay in true space; HR cursor is un-flipped above
        oend[i] = float(o.end_time)
        otype[i] = 1 if o.is_slider else (2 if o.is_spinner else 0)

    # --- next-object lookup per frame ----------------------------------
    next_idx = np.searchsorted(ot, grid, side="left").astype(np.int32)
    valid = next_idx < m
    safe = np.where(valid, next_idx, 0)
    dt_next = np.where(valid, ot[safe] - grid, 1e6).astype(np.float32)
    next_idx = np.where(valid, next_idx, -1).astype(np.int32)

    return {
        "od": np.float32(od), "ar": np.float32(ar), "cs": np.float32(cs),
        "rate": np.float32(rate), "dt": dt, "stable": np.int8(stable),
        "game_version": np.int64(getattr(frames, "game_version", 0)),
        "t": grid, "x": gx.astype(np.float32), "y": gy.astype(np.float32),
        "vx": gvx, "vy": gvy, "held": gheld, "press": press,
        "next_idx": next_idx, "dt_next": dt_next,
        "ot": ot, "ox": ox, "oy": oy, "oend": oend, "otype": otype,
    }


def extract_files(map_path: str, replay_path: str, mods_bits: int = 0) -> dict:
    bm = parse_beatmap(map_path)
    frames = load_replay(replay_path)
    return extract(bm, frames, mods_bits)


# ---------------------------------------------------------------------------
def _load_index(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    ap = argparse.ArgumentParser(description="Build dataset shards from replays.")
    ap.add_argument("--replays-index", default="data/replays/index.csv")
    ap.add_argument("--out", default="data/datasets")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--dt", type=float, default=DT_MS)
    args = ap.parse_args()

    rows = _load_index(Path(args.replays_index))
    if args.limit:
        rows = rows[: args.limit]
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    ok = fail = 0
    for i, r in enumerate(rows, 1):
        rp, mp = Path(r["replay_path"]), Path(r["map_path"])
        dst = out_root / r["beatmap_id"] / f"{r['score_id']}.npz"
        if dst.exists():
            ok += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            from pipeline.mods import acronyms_to_bits
            data = extract_files(str(mp), str(rp), acronyms_to_bits(r["mods"]))
            np.savez_compressed(dst, **data)
            ok += 1
        except Exception as e:  # noqa: BLE001
            fail += 1
            print(f"[{i}/{len(rows)}] FAIL {r['score_id']}: {type(e).__name__}: {e}")
        if i % 50 == 0:
            print(f"[{i}/{len(rows)}] ok={ok} fail={fail}", flush=True)
    print(f"done. ok={ok} fail={fail} -> {out_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
