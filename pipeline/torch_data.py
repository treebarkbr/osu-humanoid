"""Turn dataset shards into tensor batches for the three heads.

Each sample is anchored at a frame ``i`` with a fixed history window ``W`` and a
look-ahead of ``K`` upcoming object heads:

  hist (W,5) : (x, y, vx, vy, held) per past frame, normalised to [0,1]/px-ms
  ctx  (K,4) : upcoming objects relative to the cursor -> (dx, dy, dt, type)
  meta (4,)  : (od, ar, held_now, dt_next_norm)
  act  (8,)  : active-object features (slider ball / spinner phase), see
               pipeline.play_geometry -- lets the net follow sliders and orbit
               spinners; computed from a per-beatmap geometry cache

Targets:
  aim  (2,)  : cursor delta to the next frame  (dx, dy) in playfield units
  acc  (2,)  : delta *change* (acceleration) -- predicted for a momentum model
  tap  (1,)  : press-edge probability for this frame
  read (2,)  : cursor position at the *next object's* hit time (anticipation)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from pipeline.play_geometry import act_features

W = 16          # history frames
K = 6           # look-ahead objects
H = 8           # aim look-ahead horizon (frames) for the target position
POS = 512.0
TMS = 1000.0
ACT = 8         # active-object feature width
TAP_SIGMA = 1.4  # frames; soft tap target width (peaked -> decodable argmax)
TAP_RADIUS = 3   # frames each side of a press to smear the soft tap target


def _soft_tap(press: np.ndarray) -> np.ndarray:
    """Smear the 1-frame press spikes into a peaked bump so the tap net learns a
    localised peak (its argmax is then the press time) instead of a monotone
    ramp that saturates at the window edge."""
    r = np.arange(-TAP_RADIUS, TAP_RADIUS + 1, dtype=np.float64)
    ker = np.exp(-0.5 * (r / TAP_SIGMA) ** 2)
    return np.clip(np.convolve(press.astype(np.float64), ker, mode="same"), 0.0, 1.0)


def _shard_samples(d: dict, geom: dict, w: int = W, k: int = K, stride: int = 1):
    t = d["t"].astype(np.float64)
    x = d["x"].astype(np.float64)
    y = d["y"].astype(np.float64)
    vx = d["vx"].astype(np.float64)
    vy = d["vy"].astype(np.float64)
    held = d["held"].astype(np.float64)
    press = d["press"].astype(np.float64)
    nidx = d["next_idx"].astype(np.int64)
    dt_next = d["dt_next"].astype(np.float64)
    ot, ox, oy, oend, otype = d["ot"], d["ox"], d["oy"], d["oend"], d["otype"]
    od, ar = float(d["od"]), float(d["ar"])
    n = len(t)
    m = len(ot)
    if n <= w:
        return None

    idx = np.arange(w - 1, n - 1, max(1, stride))
    idx = idx[nidx[idx] >= 0]
    if len(idx) == 0:
        return None

    off = np.arange(-(w - 1), 1)
    win = idx[:, None] + off[None, :]                 # (S, w)
    hist = np.stack([x[win], y[win], vx[win], vy[win], held[win]], axis=-1)
    hist[..., 0] /= POS
    hist[..., 1] /= POS
    hist[..., 2] /= POS
    hist[..., 3] /= POS

    nid = nidx[idx]
    koff = np.arange(k)
    oid = nid[:, None] + koff[None, :]                # (S, k)
    ok = oid < m
    oids = np.where(ok, oid, 0)
    ctx = np.zeros((len(idx), k, 4), dtype=np.float64)
    ctx[..., 0] = (ox[oids] - x[idx][:, None]) / POS
    ctx[..., 1] = (oy[oids] - y[idx][:, None]) / POS
    ctx[..., 2] = (ot[oids] - t[idx][:, None]) / TMS
    ctx[..., 3] = otype[oids] / 2.0
    ctx[~ok] = 0.0

    meta = np.stack([np.full(len(idx), od / 10.0),
                     np.full(len(idx), ar / 10.0),
                     held[idx],
                     np.clip(dt_next[idx] / TMS, 0.0, 1.0)], axis=-1)

    act = act_features(t[idx], x[idx], y[idx], ot, ox, oy, oend, otype, geom)

    fut = np.clip(idx + 1, 0, n - 1)
    prv = np.clip(idx - 1, 0, n - 1)
    cur_dx, cur_dy = x[fut] - x[idx], y[fut] - y[idx]
    prv_dx, prv_dy = x[idx] - x[prv], y[idx] - y[prv]
    tgt_aim = np.stack([cur_dx, cur_dy], axis=-1)
    tgt_acc = np.stack([cur_dx - prv_dx, cur_dy - prv_dy], axis=-1)
    tgt_tap = _soft_tap(press)[idx][:, None]

    otn = ot[nid]
    j = np.clip(np.searchsorted(t, otn), 0, n - 1)
    tgt_read = np.stack([x[j] / POS, y[j] / POS], axis=-1)

    return (hist.astype(np.float32), ctx.astype(np.float32), meta.astype(np.float32),
            act.astype(np.float32),
            tgt_aim.astype(np.float32), tgt_acc.astype(np.float32),
            tgt_tap.astype(np.float32), tgt_read.astype(np.float32))


def build_samples(root: str | Path, geom_dir: str | Path | None = None,
                  max_shards: int = 0, w: int = W, k: int = K, stride: int = 1):
    root = Path(root)
    geom_dir = Path(geom_dir) if geom_dir else root.parent / "geometry"
    files = sorted(root.glob("*/*.npz"))
    if max_shards:
        files = files[:max_shards]
    parts = {name: [] for name in
             ("hist", "ctx", "meta", "act", "aim", "acc", "tap", "read")}
    groups: list[np.ndarray] = []
    geom_cache: dict[str, dict] = {}
    skipped = 0
    for si, f in enumerate(files):
        bid = f.parent.name
        if bid not in geom_cache:
            gp = geom_dir / f"{bid}.npz"
            if not gp.exists():
                geom_cache[bid] = None
            else:
                from pipeline.play_geometry import load_geometry
                geom_cache[bid] = load_geometry(gp)
        geom = geom_cache[bid]
        if geom is None:
            skipped += 1
            continue
        try:
            with np.load(f) as d:
                out = _shard_samples({key: d[key] for key in d.files}, geom, w, k, stride)
        except Exception:  # noqa: BLE001
            continue
        if out is None:
            continue
        for name, arr in zip(parts, out):
            parts[name].append(arr)
        groups.append(np.full(out[0].shape[0], si, dtype=np.int64))
    if not parts["hist"]:
        raise RuntimeError(f"no usable shards under {root}")
    if skipped:
        print(f"[torch_data] skipped {skipped} shards with no geometry", flush=True)
    out = {name: torch.from_numpy(np.concatenate(v, axis=0)) for name, v in parts.items()}
    out["group"] = torch.from_numpy(np.concatenate(groups))
    return out


def load_tensors(path: str | Path, group_offset: int = 0) -> dict:
    """Load a pre-windowed tensor shard (as written by scripts/dagger.py)."""
    with np.load(path) as d:
        out = {k: torch.from_numpy(d[k]) for k in d.files}
    if group_offset:
        out["group"] = out["group"] + group_offset
    return out


def concat(a: dict, b: dict) -> dict:
    keys = [k for k in ("hist", "ctx", "meta", "act", "aim", "acc", "tap", "read", "group") if k in a]
    return {k: torch.cat([a[k], b[k]], dim=0) for k in keys}


def split(tensors: dict, val_frac: float = 0.1, seed: int = 0):
    """Split by replay (group) so no frames from one replay appear in both sets."""
    group = tensors["group"]
    groups = torch.unique(group)
    g = torch.Generator().manual_seed(seed)
    perm = groups[torch.randperm(groups.numel(), generator=g)]
    nv = max(1, int(len(groups) * val_frac))
    val_g = set(perm[:nv].tolist())
    val_mask = torch.tensor([int(x) in val_g for x in group.tolist()], dtype=torch.bool)
    tr_mask = ~val_mask
    keep = tuple(k for k in ("hist", "ctx", "meta", "act", "aim", "acc", "tap", "read") if k in tensors)
    return ({k: tensors[k][tr_mask] for k in keep},
            {k: tensors[k][val_mask] for k in keep})
