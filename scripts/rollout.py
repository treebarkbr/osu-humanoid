"""Generate a play with the three nets.

``generate`` runs the aim/tap nets autoregressively over a map and returns the
raw cursor/key trajectory (nomod, so map time == real time here). That play can
then be:

  * judged  -- ``judge_generated`` feeds it to the headless judge;
  * rendered -- consumed by ``scripts/render.py``;
  * exported -- ``write_osr`` packs it into a real ``.osr`` for osu!lazer.

    .venv/bin/python scripts/rollout.py --maps 3 --profile average
    .venv/bin/python scripts/rollout.py --maps 1 --osr out.osr --render out.mp4
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import AimNet, TappingNet  # noqa: E402
from pipeline.beatmap import Beatmap, parse_beatmap  # noqa: E402
from pipeline.humanize import from_profile, jitter_cursor, sample_error  # noqa: E402
from pipeline.judge import judge_replay  # noqa: E402
from pipeline.play_geometry import act_features, geom_from_beatmap  # noqa: E402
from pipeline.replay import ReplayFrames  # noqa: E402
from pipeline.replay_writer import OutFrame, OutReplay, write_replay  # noqa: E402
from pipeline.torch_data import H, K, POS, TMS, W  # noqa: E402

DT = 16.0
LAZER_VERSION = 30000016
# osu!stable key bitmask: M1=1, M2=2, K1=4, K2=8. Alternating players tap
# left (M1|K1 = 5) then right (M2|K2 = 10) -- that's what real replays record.
LEFT = 0b0101
RIGHT = 0b1010
HOLD_MS = 48.0


def _ctx(ox, oy, ot, otype, i, cx, cy, t):
    c = np.zeros((K, 4), dtype=np.float32)
    for j in range(K):
        k = i + j
        if k >= len(ot):
            break
        c[j, 0] = (ox[k] - cx) / POS
        c[j, 1] = (oy[k] - cy) / POS
        c[j, 2] = (ot[k] - t) / TMS
        c[j, 3] = otype[k] / 2.0
    return c


def _hist_at(xs: np.ndarray, ys: np.ndarray, i: int) -> np.ndarray:
    """Rebuild the (W,5) history window ending at frame ``i`` from the trajectory."""
    h = np.zeros((W, 5), dtype=np.float32)
    for j in range(W):
        k = i - (W - 1) + j
        if k < 0:
            k = 0
        h[j, 0] = xs[k] / POS
        h[j, 1] = ys[k] / POS
        if k > 0:
            h[j, 2] = (xs[k] - xs[k - 1]) / DT / POS
            h[j, 3] = (ys[k] - ys[k - 1]) / DT / POS
    return h


def _act1(t, cx, cy, ot, ox, oy, oend, otype, geom):
    return act_features(np.array([t], dtype=np.float64),
                        np.array([cx], dtype=np.float64),
                        np.array([cy], dtype=np.float64),
                        ot, ox, oy, oend, otype, geom)[0]


def generate(bm: Beatmap, aim, tap, *, od, ar, rng, params, threshold=0.5, clip=60.0,
             momentum=False, vclip=60.0):
    """Return the generated play as a dict of numpy arrays + objects."""
    objs = [o for o in bm.hit_objects if o.is_clickable or o.is_spinner]
    ot = np.array([o.time for o in objs], dtype=np.float64)
    oend = np.array([o.end_time for o in objs], dtype=np.float64)
    ox = np.array([o.x for o in objs], dtype=np.float64)
    oy = np.array([o.y for o in objs], dtype=np.float64)
    otype = np.array([1 if o.is_slider else (2 if o.is_spinner else 0)
                      for o in objs], dtype=np.float64)
    geom = geom_from_beatmap(bm)

    t0 = max(0.0, ot[0] - 800.0)
    t1 = oend[-1] + 400.0
    grid = np.arange(t0, t1 + DT, DT)
    n = len(grid)

    hist = np.zeros((W, 5), dtype=np.float32)
    cx, cy = float(ox[0]), float(oy[0])
    for j in range(W):
        hist[j, 0] = cx / POS
        hist[j, 1] = cy / POS
    xs = np.zeros(n, dtype=np.float64)
    ys = np.zeros(n, dtype=np.float64)
    speeds = np.zeros(n, dtype=np.float64)
    tap_p = np.zeros(n, dtype=np.float64)

    oi = 0
    vx_c, vy_c = 0.0, 0.0          # momentum state (px/frame) when momentum=True
    for i in range(n):
        t = grid[i]
        while oi < len(ot) and ot[oi] < t:
            oi += 1
        prev = oi - 1
        held = 1.0 if (prev >= 0 and otype[prev] != 0.0 and t <= oend[prev]) else 0.0
        ctx = _ctx(ox, oy, ot, otype, oi, cx, cy, t)
        act = _act1(t, cx, cy, ot, ox, oy, oend, otype, geom)
        dt_next = (ot[oi] - t) if oi < len(ot) else 1e6
        meta = np.array([od / 10.0, ar / 10.0, held,
                         float(np.clip(dt_next / TMS, 0, 1))], dtype=np.float32)
        with torch.no_grad():
            d = aim(torch.from_numpy(hist[None]), torch.from_numpy(ctx[None]),
                    torch.from_numpy(meta[None]), torch.from_numpy(act[None])).numpy()[0]
            tp = float(torch.sigmoid(tap(torch.from_numpy(hist[None]),
                                         torch.from_numpy(ctx[None]),
                                         torch.from_numpy(meta[None]),
                                         torch.from_numpy(act[None]))).item())
        tap_p[i] = tp
        if momentum:
            # net predicts acceleration; integrate velocity (inertia -> braking)
            a = np.clip(d, -clip, clip)
            vx_c = float(np.clip(vx_c + float(a[0]), -vclip, vclip))
            vy_c = float(np.clip(vy_c + float(a[1]), -vclip, vclip))
            dx_, dy_ = vx_c, vy_c
        else:
            d = np.clip(d, -clip, clip)  # sane per-frame cap (px)
            dx_, dy_ = float(d[0]), float(d[1])
        ncx = float(np.clip(cx + dx_, 0.0, 512.0))
        ncy = float(np.clip(cy + dy_, 0.0, 384.0))
        dx_, dy_ = ncx - cx, ncy - cy   # actual applied delta (wall clamp)
        cx, cy = ncx, ncy
        if momentum:
            vx_c, vy_c = dx_, dy_        # keep integrator state consistent with reality
        xs[i], ys[i] = cx, cy
        speeds[i] = float(np.hypot(dx_, dy_) / DT)
        hist = np.roll(hist, -1, axis=0)
        hist[-1] = (cx / POS, cy / POS, (dx_ / DT) / POS, (dy_ / DT) / POS, held)

    xs, ys = jitter_cursor(xs, ys, speeds, params, rng=rng)

    press = np.zeros(n, dtype=np.uint8)
    press = np.zeros(n, dtype=np.uint8)
    for oi2 in range(len(ot)):
        lo = int(np.searchsorted(grid, ot[oi2] - 150))
        hi = int(np.searchsorted(grid, ot[oi2] + 150))
        if hi <= lo:
            continue
        i = lo + int(np.argmax(tap_p[lo:min(hi, n)]))
        if tap_p[i] >= threshold:
            sig = (params.sigma * 0.5)
            err = params.bias_ms + rng.normal(0.0, sig)
            if rng.random() < params.lapse_rate:
                err += rng.normal(0.0, params.lapse_ms)
            cell = int(np.clip(np.searchsorted(grid, grid[i] + err), 0, n - 1))
            press[cell] = 1
    keys = np.zeros(n, dtype=np.uint8)
    alt = 0
    cur = 0
    release_at = -1.0
    for i in range(n):
        t = grid[i]
        if press[i]:
            cur = LEFT if alt == 0 else RIGHT
            alt ^= 1
            release_at = t + HOLD_MS
            j = int(np.searchsorted(ot, t + 1.0)) - 1
            if 0 <= j < len(ot) and abs(ot[j] - t) <= 90.0 and otype[j] != 0.0:
                release_at = max(release_at, float(oend[j]) + 40.0)
        keys[i] = cur if t < release_at else 0

    return {"grid": grid, "x": xs, "y": ys, "keys": keys, "objs": objs,
            "ot": ot, "ox": ox, "oy": oy, "oend": oend, "otype": otype,
            "tap_p": tap_p, "geom": geom,
            "od": od, "ar": ar, "cs": bm.cs}


def to_frames(gen) -> ReplayFrames:
    grid = gen["grid"]
    return ReplayFrames(
        times=grid.astype(np.int64), x=gen["x"].astype(np.float32),
        y=gen["y"].astype(np.float32), keys=gen["keys"],
        mods=(), count300=0, count100=0, count50=0, count_miss=0,
        max_combo=0, beatmap_hash="", username="osu-humanoid",
        game_version=LAZER_VERSION,
    )


def judge_generated(bm, gen):
    return judge_replay(bm, to_frames(gen), ())


def dance_metric(bm, gen):
    """Smoothness vs a human: detour ratio (>1 = loops) and reversal rate.

    detour = cursor path length / straight-line distance between consecutive
    object heads, per segment (median). reversals = sharp (>150 deg) velocity
    direction changes per second.
    """
    objs = gen["objs"]
    ot = np.array([o.time for o in objs]); ox = np.array([o.x for o in objs]); oy = np.array([o.y for o in objs])
    xs, ys, grid = gen["x"], gen["y"], gen["grid"]
    ratios = []
    for k in range(len(objs) - 1):
        t0, t1 = ot[k], ot[k + 1]
        if t1 - t0 < 40 or t1 - t0 > 1200:
            continue
        i0, i1 = int(np.searchsorted(grid, t0)), int(np.searchsorted(grid, t1))
        if i1 - i0 < 2:
            continue
        plen = float(np.sum(np.hypot(np.diff(xs[i0:i1 + 1]), np.diff(ys[i0:i1 + 1]))))
        slen = float(np.hypot(ox[k + 1] - ox[k], oy[k + 1] - oy[k]))
        if slen > 20:
            ratios.append(plen / slen)
    detour = float(np.median(ratios)) if ratios else float("nan")
    vx, vy = np.gradient(xs), np.gradient(ys)
    sp = np.hypot(vx, vy)
    m = sp > 0.3
    vx, vy = vx[m], vy[m]
    rev = 0.0
    if len(vx) > 2:
        cos = (vx[1:] * vx[:-1] + vy[1:] * vy[:-1]) / (np.hypot(vx[1:], vy[1:]) * np.hypot(vx[:-1], vy[:-1]) + 1e-9)
        ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        rev = float(np.sum(ang > 150) / max((grid[-1] - grid[0]) / 1000.0, 1e-9))
    return detour, rev


def beatmap_md5(path: str | Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def write_osr(bm: Beatmap, gen, path: str | Path, *, map_path: str | Path,
              username: str = "osu-humanoid") -> Path:
    """Pack a generated play into a real .osr (counts filled from the judge)."""
    rep = judge_generated(bm, gen)
    c = rep.counts()
    frames = [OutFrame(int(t), float(x), float(y), int(k))
              for t, x, y, k in zip(gen["grid"], gen["x"], gen["y"], gen["keys"])]
    out = OutReplay(
        beatmap_hash=beatmap_md5(map_path), username=username,
        mods=0, game_version=LAZER_VERSION,
        count300=c["300"], count100=c["100"], count50=c["50"], count_miss=c["miss"],
        max_combo=c["300"] + c["100"] + c["50"],
        perfect=c["miss"] == 0 and c["100"] == 0 and c["50"] == 0,
        frames=frames,
    )
    return write_replay(out, path)


def _load(ckpt: str | Path):
    ck = Path(ckpt)
    load = lambda p: torch.load(p, map_location="cpu")  # noqa: E731

    def _act_of(sd):
        return max(0, sd["net.0.weight"].shape[1] - (W * 5 + K * 4 + 4))

    asd = load(ck / "aim.pt")
    aim = AimNet(act=_act_of(asd)); aim.load_state_dict(asd); aim.eval()
    tsd = load(ck / "tap.pt")
    tap = TappingNet(act=_act_of(tsd)); tap.load_state_dict(tsd); tap.eval()
    return aim, tap


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints")
    ap.add_argument("--maps", type=int, default=5)
    ap.add_argument("--profile", default="average")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--clip", type=float, default=60.0, help="per-frame cursor delta cap (px)")
    ap.add_argument("--momentum", action="store_true", help="treat aim output as acceleration and integrate")
    ap.add_argument("--vclip", type=float, default=60.0, help="velocity cap (px/frame) in momentum mode")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    params = from_profile(args.profile)
    aim, tap = _load(args.ckpt)

    dets, revs, stops = [], [], []
    for p in sorted(Path("data/beatmaps").glob("*.osu"))[: args.maps]:
        bm = parse_beatmap(p)
        gen = generate(bm, aim, tap, od=bm.od, ar=bm.ar, rng=rng,
                       params=params, threshold=args.threshold, clip=args.clip,
                       momentum=args.momentum, vclip=args.vclip)
        rep = judge_generated(bm, gen)
        det, rev = dance_metric(bm, gen)
        v = np.hypot(np.diff(gen["x"]), np.diff(gen["y"]))
        dets.append(det); revs.append(rev); stops.append(float(np.mean(v < 1.0)))
        print(f"{p.stem} {bm.version[:24]:24s} objs={len(rep.results):4d}  {rep.summary()} "
              f"detour={det:.3f} rev/s={rev:.1f} stop%={np.mean(v < 1.0) * 100:.1f}")
    if dets:
        print(f"MEAN detour={np.nanmean(dets):.3f} rev/s={np.mean(revs):.1f} "
              f"stop%={np.mean(stops) * 100:.1f} momentum={args.momentum}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
