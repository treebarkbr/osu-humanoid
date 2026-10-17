"""Train the *aim* head on the gfx902 iGPU via OpenCL (pipeline.ocl).

    .venv/bin/python scripts/train_ocl.py --epochs 12 --batch 4096 --out checkpoints_ocl

The aim head is reactive: it maps the present state to *this frame's* cursor
delta (px), so the decoding is a plain integration -- no look-ahead targets,
no low-pass.  Weights are saved as a torch state_dict so ``scripts/rollout.py``
/ ``scripts/watch.py`` keep loading them unchanged.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import ocl  # noqa: E402
from pipeline.opt import warmup_cosine  # noqa: E402
from pipeline.torch_data import K, W, build_samples, load_tensors, split  # noqa: E402

INDIM = W * 5 + K * 4 + 4
HID = 256
DIMS = [INDIM, HID, HID, 2]
ACTS = ["relu", "relu", "none"]


def _flatten(d) -> np.ndarray:
    return np.concatenate([d["hist"].reshape(d["hist"].shape[0], -1).numpy(),
                           d["ctx"].reshape(d["ctx"].shape[0], -1).numpy(),
                           d["meta"].numpy()], axis=1).astype(np.float32)


def _smoothl1(pred, tgt, beta=1.0):
    d = np.abs(pred - tgt)
    return float(np.where(d < beta, 0.5 * d * d / beta, d - 0.5 * beta).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default="data/datasets")
    ap.add_argument("--out", default="checkpoints_ocl")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--limit-shards", type=int, default=0)
    ap.add_argument("--dagger", default="", help="comma-separated dagger .npz to add to aim train set")
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--min-lr-factor", type=float, default=0.05)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--loss-every", type=int, default=32,
                    help="read back train loss every N steps (0 = only val)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--copy-from", default="checkpoints_fix",
                    help="dir to copy tap.pt/read.pt from")
    ap.add_argument("--notify-cmd", default="")
    args = ap.parse_args()

    ocl._setup()
    print("device:", ocl.device_name(), flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for s in ("_DONE", "_done.json"):
        try:
            (out / s).unlink()
        except FileNotFoundError:
            pass

    print("building samples ...", flush=True)
    t0 = time.time()
    data = build_samples(args.dataset_dir, max_shards=args.limit_shards, stride=args.stride)
    tr, va = split(data, args.val_frac)
    Xtr, Ytr = _flatten(tr), tr["aim"].numpy().astype(np.float32)
    Xva, Yva = _flatten(va), va["aim"].numpy().astype(np.float32)
    if args.dagger:
        for p in args.dagger.split(","):
            dd = load_tensors(p)
            Xtr = np.concatenate([Xtr, _flatten(dd)], axis=0)
            Ytr = np.concatenate([Ytr, dd["aim"].numpy().astype(np.float32)], axis=0)
    n = Xtr.shape[0]
    print(f"samples: train={n} val={Xva.shape[0]} ({time.time()-t0:.1f}s build) dim={INDIM}", flush=True)

    net = ocl.MLP(DIMS, ACTS, seed=args.seed)
    q = ocl._q
    b = args.batch
    Xbuf = ocl.empty(b * INDIM)
    Ybuf = ocl.empty(b * 2)
    steps_per_epoch = (n + b - 1) // b
    total = steps_per_epoch * args.epochs
    sched = warmup_cosine(args.warmup, total, args.min_lr_factor)
    rng = np.random.default_rng(args.seed)
    Xva_buf = ocl.up(Xva)
    best = float("inf")
    best_state = None
    gstep = 0
    for ep in range(1, args.epochs + 1):
        perm = rng.permutation(n)
        tot = 0.0
        cnt = 0
        te = time.time()
        for i in range(0, n, b):
            idx = perm[i:i + b]
            m = len(idx)
            cl = ocl.cl
            cl.enqueue_copy(q, Xbuf, np.ascontiguousarray(Xtr[idx]))
            cl.enqueue_copy(q, Ybuf, np.ascontiguousarray(Ytr[idx]))
            ocl_out = net.forward(Xbuf, m)
            gpred = ocl.empty(m * 2)
            ocl.smoothl1_grad(ocl_out, Ybuf, gpred, m * 2, 1.0, 1.0 / m)
            net.backward(gpred, m)
            net.step(args.lr * sched(gstep), wd=args.wd)
            gstep += 1
            if args.loss_every and (gstep % args.loss_every == 0):
                pred = ocl.down(ocl_out, (m, 2))
                tot += _smoothl1(pred, Ytr[idx]) * m
                cnt += m
        # val
        pv = net.forward(Xva_buf, Xva.shape[0])
        val = _smoothl1(ocl.down(pv, (Xva.shape[0], 2)), Yva)
        mark = ""
        if val < best:
            best = val
            best_state = net.state()
            mark = " *"
        print(f"ep{ep:2d} train={tot/max(cnt,1):.5f} val={val:.5f} "
              f"({time.time()-te:.1f}s){mark}", flush=True)

    # ---- save as torch AimNet state_dict (Linear stores (out,in): transpose)
    sd = {}
    dims = DIMS
    for l in range(len(ACTS)):
        Wt = np.ascontiguousarray(best_state["W"][l].T)  # (out,in)
        sd[f"net.{2*l}.weight"] = torch.from_numpy(Wt.copy())
        sd[f"net.{2*l}.bias"] = torch.from_numpy(best_state["b"][l].copy())
    torch.save(sd, out / "aim.pt")

    # copy tap/read (unchanged pipelines)
    src = Path(args.copy_from)
    for name in ("tap.pt", "read.pt"):
        if (src / name).exists() and not (out / name).exists():
            shutil.copy(src / name, out / name)

    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    (out / "_done.json").write_text(json.dumps({"status": "ok", "ts": stamp, "best": best}, indent=2))
    (out / "_DONE").write_text(f"ok {stamp} aim_val={best:.5f}\n")
    print(f"TRAINING DONE status=ok aim_val={best:.5f} dims={dims} -> {out/'aim.pt'}", flush=True)
    if args.notify_cmd:
        subprocess.run(args.notify_cmd, shell=True, check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
