"""Light CPU trainer for the three heads (reading, aim, tapping).

    .venv/bin/python train.py --epochs 6 --batch 4096
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import AimNet, ReadingNet, TappingNet  # noqa: E402
from pipeline.opt import EMA, MuonAdamW, focal_bce, warmup_cosine  # noqa: E402
from pipeline.torch_data import (build_samples, concat, load_tensors,  # noqa: E402
                                 split)


def _batches(n: int, bs: int, perm: torch.Tensor):
    for i in range(0, n, bs):
        yield perm[i:i + bs]


def _make_opt(model, args):
    if args.optim == "muon":
        return MuonAdamW(model, lr=args.lr, muon_lr=args.muon_lr,
                         weight_decay=args.weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay)


def train_head(name, model, tr, va, *, kind, args, out_dir, pos_weight=None):
    epochs, bs = args.epochs, args.batch
    opt = _make_opt(model, args)
    ema = EMA(model, decay=args.ema_decay, warmup=args.ema_warmup) if args.ema else None
    if kind == "bce":
        if args.focal_gamma > 0:
            lossf = lambda o, t: focal_bce(o, t, gamma=args.focal_gamma,
                                           alpha=args.focal_alpha,
                                           label_smooth=args.label_smooth)
        else:
            lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        lossf = nn.SmoothL1Loss()

    n = tr["hist"].shape[0]
    steps_per_epoch = (n + bs - 1) // bs
    total_steps = steps_per_epoch * epochs
    sched = warmup_cosine(args.warmup, total_steps, args.min_lr_factor)
    g = torch.Generator().manual_seed(0)
    best = float("inf")
    step = 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=g)
        tot = cnt = 0.0
        t0 = time.time()
        for b in _batches(n, bs, perm):
            factor = sched(step) if args.schedule == "warmup_cosine" else 1.0
            if args.optim == "muon":
                opt.set_scale(factor)
            else:
                for grp in opt.param_groups:
                    grp["lr"] = args.lr * factor
            opt.zero_grad()
            out = model(tr["hist"][b], tr["ctx"][b], tr["meta"][b])
            loss = lossf(out, tr[_KEY[name]][b])
            loss.backward()
            opt.step()
            if ema:
                ema.update()
            tot += loss.item()
            cnt += 1
            step += 1
        model.eval()
        if ema:
            ema.copy_to(model)
        with torch.no_grad():
            out = model(va["hist"], va["ctx"], va["meta"])
            vloss = float(lossf(out, va[_KEY[name]]))
        if ema:
            ema.restore()
        mark = ""
        if vloss < best:
            best = vloss
            saved = dict(model.state_dict())
            if ema:
                ema.copy_to(model)
                saved = dict(model.state_dict())
                ema.restore()
            torch.save(saved, out_dir / f"{name}.pt")
            mark = " *"
        print(f"  [{name}] ep{ep} train={tot / max(cnt, 1):.5f} "
              f"val={vloss:.5f} ({time.time() - t0:.1f}s){mark}", flush=True)
    return best


_KEY = {"aim": "aim", "tap": "tap", "read": "read"}


def _signal_done(out_dir: Path, status: str, bests: dict, notify_cmd: str = "") -> None:
    """Drop a pollable completion marker (and optionally run a notify command)."""
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    payload = {"status": status, "ts": stamp, "best": bests}
    (out_dir / "_done.json").write_text(json.dumps(payload, indent=2))
    (out_dir / "_DONE").write_text(f"{status} {stamp}\n")
    line = f"TRAINING DONE status={status} best={bests} at {stamp}"
    print(line, flush=True)
    if notify_cmd:
        try:
            subprocess.run(notify_cmd, shell=True, check=False)
        except Exception as e:  # noqa: BLE001
            print(f"notify-cmd failed: {e}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default="data/datasets")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit-shards", type=int, default=0)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--optim", choices=["adam", "muon"], default="adam")
    ap.add_argument("--muon-lr", type=float, default=0.02)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--schedule", choices=["const", "warmup_cosine"], default="const")
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--min-lr-factor", type=float, default=0.05)
    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--ema-warmup", type=int, default=0)
    ap.add_argument("--focal-gamma", type=float, default=0.0)
    ap.add_argument("--focal-alpha", type=float, default=0.5)
    ap.add_argument("--label-smooth", type=float, default=0.0)
    ap.add_argument("--dagger", default="", help="dir of DAgger .npz shards (augments aim)")
    ap.add_argument("--notify-cmd", default="",
                    help="shell command to run when training finishes")
    args = ap.parse_args()

    torch.set_num_threads(max(1, torch.get_num_threads()))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("_DONE", "_done.json"):
        try:
            (out_dir / stale).unlink()
        except FileNotFoundError:
            pass

    print("building samples ...", flush=True)
    t0 = time.time()
    data = build_samples(args.dataset_dir, max_shards=args.limit_shards,
                         stride=args.stride)
    tr, va = split(data, args.val_frac)
    n = tr["hist"].shape[0]
    print(f"samples: train={n} val={va['hist'].shape[0]} "
          f"({time.time() - t0:.1f}s build)", flush=True)

    pos = float(tr["tap"].sum())
    pos_weight = torch.tensor([(n - pos) / max(pos, 1.0)])

    tr_by_head = {"aim": tr, "tap": tr, "read": tr}
    if args.dagger:
        import glob
        files = sorted(glob.glob(str(Path(args.dagger) / "*.npz")))
        if files:
            dag = None
            for i, f in enumerate(files):
                d = load_tensors(f, group_offset=1_000_000 + i * 100_000)
                dag = d if dag is None else concat(dag, d)
            tr_by_head = {"aim": concat(tr, dag), "tap": tr, "read": tr}
            print(f"dagger: +{dag['hist'].shape[0]} aim samples from "
                  f"{len(files)} file(s)", flush=True)

    cfg = [
        ("aim", AimNet(), "l1", None),
        ("tap", TappingNet(), "bce", pos_weight),
        ("read", ReadingNet(), "l1", None),
    ]
    bests: dict = {}
    try:
        for name, model, kind, pw in cfg:
            print(f"training {name} ({sum(p.numel() for p in model.parameters())} params)",
                  flush=True)
            best = train_head(name, model, tr_by_head[name], va, kind=kind, args=args,
                              out_dir=out_dir, pos_weight=pw)
            bests[name] = best
            print(f"  best {name} val={best:.5f} -> {out_dir / (name + '.pt')}", flush=True)
    except Exception:
        _signal_done(out_dir, "error", bests, args.notify_cmd)
        raise
    _signal_done(out_dir, "ok", bests, args.notify_cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
