"""DAgger: roll the aim net out on many maps *in parallel*, relabel its own
states with an expert human cursor, and save compatible samples for the next
training round.

Speed: the old version ran the aim net one frame at a time (batch-of-1) in a
Python loop -- ~2 s/replay, almost all framework overhead.  We now simulate
every map in lock-step and issue one *batched* forward per timestep (maps
grouped by length so padding stays low).  Same math, ~100x fewer forwards.

Expert labels are *corrective*: aim target = expert_next_pos - model_pos (not
expert_next - expert_pos).  In teacher forcing that is exactly the human delta;
off-policy (model drifted) it becomes a strong restorative signal, which is what
kills the compounding drift.

Why replays are valid experts for every map: features are in *map time*, so
every replay (DT/HT/HR included, HR already un-flipped) matches every map.

    .venv/bin/python scripts/dagger.py --maps 700 --out data/dagger/iter1.npz
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import AimNet  # noqa: E402
from pipeline.beatmap import Beatmap, parse_beatmap  # noqa: E402
from pipeline.judge import _is_legacy_version, press_events  # noqa: E402
from pipeline.mods import acronyms_to_bits, clock_rate, effective_difficulty  # noqa: E402
from pipeline.play_geometry import act_features, geom_from_beatmap  # noqa: E402
from pipeline.replay import load_replay  # noqa: E402
from pipeline.torch_data import K, POS, TMS, W, _soft_tap  # noqa: E402

DT = 16.0
PLAYFIELD_H = 384.0


def _objects(bm: Beatmap):
    objs = [o for o in bm.hit_objects if o.is_clickable or o.is_spinner]
    ot = np.array([o.time for o in objs], dtype=np.float64)
    ox = np.array([o.x for o in objs], dtype=np.float64)
    oy = np.array([o.y for o in objs], dtype=np.float64)  # true space
    oend = np.array([o.end_time for o in objs], dtype=np.float64)
    otype = np.array([1 if o.is_slider else (2 if o.is_spinner else 0)
                      for o in objs], dtype=np.float64)
    return ot, ox, oy, oend, otype


def _spec(row, bm: Beatmap):
    bits = acronyms_to_bits(row["mods"]) if row.get("mods") else 0
    rate = clock_rate(bits)
    flip = bool(bits & 16)
    _cs, ar, od, _hp = effective_difficulty(bm.cs, bm.ar, bm.od, bm.hp, bits)
    ot, ox, oy, oend, otype = _objects(bm)
    return {"ot": ot, "ox": ox, "oy": oy, "oend": oend, "otype": otype,
            "od": od, "ar": ar, "rate": rate, "flip": flip,
            "geom": geom_from_beatmap(bm)}


def _traj_len(s):
    t0 = max(0.0, s["ot"][0] - 800.0)
    t1 = s["oend"][-1] + 400.0
    return t0, int(np.floor((t1 - t0) / DT)) + 1


def batch_model_trajs(specs, aim, act_dim=0, chunk=512):
    """Simulate all maps at once.  Returns list of (grid, mx, my) per spec."""
    m_total = len(specs)
    bounds = [_traj_len(s) for s in specs]
    lens = np.array([b[1] for b in bounds])
    order = np.argsort(lens)
    out = [None] * m_total

    # Cursor-*independent* part of the act features (dims 0:6): dims 0:4 are the
    # target positions, 4 is kind, 5 is progress -- none depend on the cursor, so
    # we precompute them once per map.  dims 6:7 (spinner sin/cos) do depend on
    # the cursor and are rebuilt each step from the precomputed spinner centre.
    TAB = [None] * m_total
    if act_dim > 0:
        for i, s in enumerate(specs):
            tt = bounds[i][0] + np.arange(lens[i]) * DT
            zero = np.zeros(len(tt))
            t8 = act_features(tt, zero, zero, s["ot"], s["ox"], s["oy"],
                              s["oend"], s["otype"], s["geom"])
            TAB[i] = t8[:, :6].astype(np.float64)

    for c0 in range(0, m_total, chunk):
        ids = order[c0:c0 + chunk]
        C = len(ids)
        n = lens[ids]
        nmax = int(n.max())
        nobj = np.array([len(specs[i]["ot"]) for i in ids])
        mo = int(nobj.max())

        OX = np.zeros((C, mo)); OY = np.zeros((C, mo))
        OT = np.full((C, mo), np.inf); OE = np.zeros((C, mo)); OTY = np.zeros((C, mo))
        OI = np.zeros((C, nmax), dtype=np.int64)
        tt0 = np.empty(C); cx0 = np.empty(C); cy0 = np.empty(C)
        od = np.empty(C); ar = np.empty(C)
        T0 = T1 = T2 = T3 = T4 = T5 = SPX = SPY = None
        if act_dim > 0:
            T0 = np.zeros((C, nmax)); T1 = np.zeros((C, nmax))
            T2 = np.zeros((C, nmax)); T3 = np.zeros((C, nmax))
            T4 = np.zeros((C, nmax)); T5 = np.zeros((C, nmax))
            SPX = np.zeros((C, nmax)); SPY = np.zeros((C, nmax))
        for r, i in enumerate(ids):
            s = specs[i]
            m = nobj[r]
            OX[r, :m] = s["ox"]; OY[r, :m] = s["oy"]; OT[r, :m] = s["ot"]
            OE[r, :m] = s["oend"]; OTY[r, :m] = s["otype"]
            tt0[r] = bounds[i][0]
            cx0[r], cy0[r] = s["ox"][0], s["oy"][0]
            od[r], ar[r] = s["od"], s["ar"]
            OI[r, :n[r]] = np.searchsorted(s["ot"], tt0[r] + np.arange(n[r]) * DT, side="left")
            if act_dim > 0:
                tb = TAB[i]
                T0[r, :n[r]] = tb[:, 0]; T1[r, :n[r]] = tb[:, 1]
                T2[r, :n[r]] = tb[:, 2]; T3[r, :n[r]] = tb[:, 3]
                T4[r, :n[r]] = tb[:, 4]; T5[r, :n[r]] = tb[:, 5]
                SPX[r, :n[r]] = tb[:, 0] * POS
                SPY[r, :n[r]] = tb[:, 1] * POS

        cx = cx0.astype(np.float64).copy()
        cy = cy0.astype(np.float64).copy()
        hist = np.zeros((C, W, 5), dtype=np.float32)
        hist[:, :, 0] = (cx / POS)[:, None]
        hist[:, :, 1] = (cy / POS)[:, None]
        MX = np.zeros((C, nmax)); MY = np.zeros((C, nmax))
        koff = np.arange(K)

        aim.eval()
        for s_i in range(nmax):
            act = np.where(s_i < n)[0]
            if act.size == 0:
                break
            t = tt0[act] + s_i * DT
            oiA = OI[act, s_i]
            kk = oiA[:, None] + koff[None, :]                     # (A,K)
            inb = kk < mo
            kkc = np.clip(kk, 0, mo - 1)
            otg = OT[act[:, None], kkc]
            obj_ok = inb & np.isfinite(otg)

            a2 = act[:, None]
            ctx = np.zeros((act.size, K, 4), dtype=np.float32)
            ctx[..., 0] = np.where(obj_ok, (OX[a2, kkc] - cx[act][:, None]) / POS, 0.0)
            ctx[..., 1] = np.where(obj_ok, (OY[a2, kkc] - cy[act][:, None]) / POS, 0.0)
            ctx[..., 2] = np.where(obj_ok, (otg - t[:, None]) / TMS, 0.0)
            ctx[..., 3] = np.where(obj_ok, OTY[a2, kkc] / 2.0, 0.0)

            dtnext = np.full(act.size, 1e6)
            gn = oiA < mo
            dtnext[gn] = OT[act[gn], np.clip(oiA[gn], 0, mo - 1)] - t[gn]

            prev = np.clip(oiA - 1, 0, mo - 1)
            held = np.where((oiA > 0) & np.isfinite(OT[act, prev]) &
                            (OTY[act, prev] != 0.0) & (t <= OE[act, prev]), 1.0, 0.0)

            meta = np.stack([od[act] / 10.0, ar[act] / 10.0, held,
                             np.clip(dtnext / TMS, 0.0, 1.0)], axis=-1).astype(np.float32)

            with torch.no_grad():
                if act_dim > 0:
                    cxa = cx[act]; cya = cy[act]
                    k4 = T4[act, s_i]
                    is_sp = k4 > 0.75
                    ang = np.arctan2(cya - SPY[act, s_i], cxa - SPX[act, s_i])
                    act_t = np.stack([
                        T0[act, s_i] - cxa / POS,
                        T1[act, s_i] - cya / POS,
                        T2[act, s_i] - cxa / POS,
                        T3[act, s_i] - cya / POS,
                        k4, T5[act, s_i],
                        np.where(is_sp, np.sin(ang), 0.0),
                        np.where(is_sp, np.cos(ang), 0.0),
                    ], axis=-1).astype(np.float32)
                    d = aim(torch.from_numpy(hist[act]),
                            torch.from_numpy(ctx),
                            torch.from_numpy(meta),
                            torch.from_numpy(act_t)).numpy()
                else:
                    d = aim(torch.from_numpy(hist[act]),
                            torch.from_numpy(ctx),
                            torch.from_numpy(meta)).numpy()
            d = np.clip(d, -60.0, 60.0)
            old_cx = cx[act].copy(); old_cy = cy[act].copy()
            cx[act] = np.clip(cx[act] + d[:, 0], 0.0, 512.0)
            cy[act] = np.clip(cy[act] + d[:, 1], 0.0, 384.0)
            adx = cx[act] - old_cx; ady = cy[act] - old_cy
            MX[act, s_i] = cx[act]; MY[act, s_i] = cy[act]
            h = np.roll(hist[act], -1, axis=1)
            h[:, -1, 0] = cx[act] / POS
            h[:, -1, 1] = cy[act] / POS
            h[:, -1, 2] = (adx / DT) / POS
            h[:, -1, 3] = (ady / DT) / POS
            h[:, -1, 4] = held
            hist[act] = h

        for r, i in enumerate(ids):
            grid = tt0[r] + np.arange(n[r]) * DT
            out[i] = (grid, MX[r, :n[r]], MY[r, :n[r]])
    return out


def _expert(fr, grid, flip, rate):
    stable = _is_legacy_version(fr)
    ft = fr.times.astype(np.float64) * (1.0 if stable else rate)
    fx = fr.x.astype(np.float64)
    fy = fr.y.astype(np.float64)
    if flip:
        fy = PLAYFIELD_H - fy
    ex = np.interp(grid, ft, fx)
    ey = np.interp(grid, ft, fy)
    press = np.zeros(len(grid), dtype=np.uint8)
    _pi, ptimes, _pm = press_events(fr)
    ptm = ptimes.astype(np.float64) * (1.0 if stable else rate)
    cell = np.searchsorted(grid, ptm, side="right") - 1
    cell = cell[(cell >= 0) & (cell < len(grid))]
    press[cell] = 1
    return ex, ey, press


def _ctx(ox, oy, ot, otype, rows, cx, cy, t):
    """(len(rows), K, 4) upcoming objects relative to the model cursor."""
    c = np.zeros((len(rows), K, 4), dtype=np.float64)
    oid = rows[:, None] + np.arange(K)[None, :]
    ok = oid < len(ot)
    oids = np.where(ok, oid, 0)
    c[..., 0] = np.where(ok, (ox[oids] - cx[:, None]) / POS, 0.0)
    c[..., 1] = np.where(ok, (oy[oids] - cy[:, None]) / POS, 0.0)
    c[..., 2] = np.where(ok, (ot[oids] - t[:, None]) / TMS, 0.0)
    c[..., 3] = np.where(ok, otype[oids] / 2.0, 0.0)
    return c


def _samples(spec, traj, fr, stride):
    grid, mx, my = traj
    ot, ox, oy, oend, otype = spec["ot"], spec["ox"], spec["oy"], spec["oend"], spec["otype"]
    od, ar = spec["od"], spec["ar"]
    ex, ey, press = _expert(fr, grid, spec["flip"], spec["rate"])

    n = len(grid)
    if n <= W:
        return None
    mvx = np.gradient(mx, DT)
    mvy = np.gradient(my, DT)

    nidx = np.searchsorted(ot, grid, side="left").astype(np.int64)
    valid = nidx < len(ot)
    dt_next = np.where(valid, ot[np.where(valid, nidx, 0)] - grid, 1e6)

    # held is a deterministic function of the object schedule (same rule live_sim
    # uses), so it is valid for the model's own trajectory even without taps.
    prev_i = np.clip(nidx - 1, 0, max(len(ot) - 1, 0))
    held = np.where((nidx > 0) & (otype[prev_i] != 0.0) & (grid <= oend[prev_i]),
                    1.0, 0.0) if len(ot) else np.zeros(n)

    idx = np.arange(W - 1, n - 1, max(1, stride))
    idx = idx[nidx[idx] < len(ot)]
    if len(idx) == 0:
        return None

    off = np.arange(-(W - 1), 1)
    win = idx[:, None] + off[None, :]
    hist = np.stack([mx[win], my[win], mvx[win], mvy[win], held[win]], axis=-1)
    hist[..., 0] /= POS
    hist[..., 1] /= POS
    hist[..., 2] /= POS
    hist[..., 3] /= POS

    cx, cy = mx[idx], my[idx]
    t = grid[idx]
    ctx = _ctx(ox, oy, ot, otype, nidx[idx], cx, cy, t)
    meta = np.stack([np.full(len(idx), od / 10.0), np.full(len(idx), ar / 10.0),
                     held[idx], np.clip(dt_next[idx] / TMS, 0, 1)], axis=-1)
    act = act_features(t, cx, cy, ot, ox, oy, oend, otype, spec["geom"])

    fut = np.clip(idx + 1, 0, n - 1)
    prv = np.clip(idx - 1, 0, n - 1)
    cur_dx, cur_dy = ex[fut] - ex[idx], ey[fut] - ey[idx]
    prv_dx, prv_dy = ex[idx] - ex[prv], ey[idx] - ey[prv]
    aim_t = np.clip(np.stack([ex[fut] - cx, ey[fut] - cy], axis=-1), -60.0, 60.0)
    acc_t = np.stack([cur_dx - prv_dx, cur_dy - prv_dy], axis=-1)
    tap_t = _soft_tap(press)[idx][:, None].astype(np.float64)
    otn = ot[nidx[idx]]
    j = np.clip(np.searchsorted(grid, otn), 0, n - 1)
    read_t = np.stack([ex[j] / POS, ey[j] / POS], axis=-1)

    return (hist.astype(np.float32), ctx.astype(np.float32), meta.astype(np.float32),
            act.astype(np.float32), aim_t.astype(np.float32), acc_t.astype(np.float32),
            tap_t.astype(np.float32), read_t.astype(np.float32))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints")
    ap.add_argument("--maps", type=int, default=300, help="max replays to use")
    ap.add_argument("--out", default="data/dagger/iter0.npz")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--replays-index", default="data/replays/index.csv")
    args = ap.parse_args()

    aim = AimNet()
    sd = torch.load(Path(args.ckpt) / "aim.pt", map_location="cpu")
    act_dim = int(sd["net.0.weight"].shape[1]) - (W * 5 + K * 4 + 4)
    if act_dim < 0:
        act_dim = 0
    aim = AimNet(act=act_dim)
    aim.load_state_dict(sd)
    aim.eval()
    print(f"checkpoint {args.ckpt}: act_dim={act_dim}", flush=True)

    rows = list(csv.DictReader(open(args.replays_index)))
    if args.maps:
        rows = rows[: args.maps]

    t_start = time.time()
    bm_cache: dict = {}
    specs, valid_rows = [], []
    for r in rows:
        mp = r["map_path"]
        try:
            if mp not in bm_cache:
                bm_cache[mp] = parse_beatmap(mp)
            s = _spec(r, bm_cache[mp])
        except Exception:  # noqa: BLE001
            continue
        if len(s["ot"]) == 0:
            continue
        specs.append(s)
        valid_rows.append(r)
    print(f"parsed {len(specs)} maps in {time.time()-t_start:.1f}s", flush=True)

    t1 = time.time()
    trajs = batch_model_trajs(specs, aim, act_dim=act_dim, chunk=args.chunk)
    print(f"batched rollout of {len(trajs)} maps in {time.time()-t1:.1f}s "
          f"({len(trajs)/max(time.time()-t1,1e-9):.0f} rep/s)", flush=True)

    parts = {k: [] for k in ("hist", "ctx", "meta", "act", "aim", "acc", "tap", "read")}
    groups = []
    ok = fail = 0
    for gi, (r, s, tr) in enumerate(zip(valid_rows, specs, trajs)):
        try:
            fr = load_replay(r["replay_path"])
            out = _samples(s, tr, fr, args.stride)
        except Exception:  # noqa: BLE001
            fail += 1
            continue
        if out is None:
            continue
        for name, arr in zip(parts, out):
            parts[name].append(arr)
        groups.append(np.full(out[0].shape[0], gi, dtype=np.int64))
        ok += 1
        if ok % max(1, args.log_every) == 0 or gi == len(valid_rows) - 1:
            n_samp = sum(a.shape[0] for a in parts["hist"])
            el = time.time() - t_start
            print(f"  [{gi + 1}/{len(valid_rows)}] ok={ok} fail={fail} "
                  f"samples={n_samp} ({el:.0f}s)", flush=True)

    if not parts["hist"]:
        print("no dagger samples produced")
        return 1
    data = {name: np.concatenate(v, axis=0) for name, v in parts.items()}
    data["group"] = np.concatenate(groups)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    np.savez(outp, **data)
    print(f"done: {ok} replays ok, {fail} fail, {len(data['hist'])} samples -> {outp} "
          f"({time.time()-t_start:.0f}s total)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
