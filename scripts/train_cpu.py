"""Train the *aim* head on the CPU with torch (keeps the display GPU free).

The Vega 11 in this box is an APU, so it also drives the desktop -- hammering it
starves the compositor.  This trainer keeps all heavy math on the CPU cores;
the net is tiny (108->256->256->2) so it's basically as fast, and the desktop
stays smooth.

Mirrors scripts/train_ocl.py's interface: base shards (data/datasets) plus any
``--dagger`` npz, group-wise train/val split, SmoothL1, Adam + warmup-cosine,
and a torch ``state_dict`` saved for scripts/rollout.py / scripts/watch.py.

    .venv/bin/python scripts/train_cpu.py --epochs 30 --dagger data/dagger/iter1.npz --out checkpoints
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import AimNet  # noqa: E402
from pipeline.opt import warmup_cosine  # noqa: E402
from pipeline.torch_data import POS, build_samples, concat, load_tensors, split  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default="data/datasets")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--limit-shards", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--min-lr-factor", type=float, default=0.05)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--gamma", type=float, default=0.0, help="SmoothL1 beta")
    ap.add_argument("--threads", type=int, default=0, help="0 = cpu_count-1 (leave a core for the desktop)")
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dagger", default="", help="comma-separated dagger .npz to add to the aim train set")
    ap.add_argument("--augment-frac", type=float, default=0.5,
                    help="fraction of base aim samples to perturb into off-policy states")
    ap.add_argument("--augment-sigma", type=float, default=12.0, help="perturbation offset std (px)")
    ap.add_argument("--augment-max", type=float, default=50.0, help="perturbation offset clip (px)")
    ap.add_argument("--augment-rounds", type=int, default=1, help="independent perturbed copies of the base set")
    ap.add_argument("--copy-from", default="checkpoints_fix", help="dir to copy tap.pt/read.pt from")
    ap.add_argument("--notify-cmd", default="")
    args = ap.parse_args()

    threads = args.threads or max(1, (os.cpu_count() or 2) - 1)
    torch.set_num_threads(threads)
    torch.manual_seed(args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for s in ("_DONE", "_done.json"):
        try:
            (out / s).unlink()
        except FileNotFoundError:
            pass

    print(f"threads={threads}", flush=True)
    print("building samples ...", flush=True)
    t0 = time.time()
    data = build_samples(args.dataset_dir, max_shards=args.limit_shards, stride=args.stride)
    tr, va = split(data, args.val_frac, seed=args.seed)
    if args.dagger:
        for i, p in enumerate(args.dagger.split(",")):
            dd = load_tensors(p, group_offset=10_000 + i * 1_000_000)
            tr = concat(tr, dd)
    Htr, Ctr, Mtr, Ytr = tr["hist"].contiguous(), tr["ctx"].contiguous(), tr["meta"].contiguous(), tr["aim"].contiguous()
    Hva, Cva, Mva, Yva = va["hist"].contiguous(), va["ctx"].contiguous(), va["meta"].contiguous(), va["aim"].contiguous()
    if args.augment_rounds > 0 and args.augment_frac > 0:
        Htr, Ctr, Mtr, Ytr = Htr.clone(), Ctr.clone(), Mtr.clone(), Ytr.clone()
        g2 = torch.Generator().manual_seed(args.seed + 7)
        n0 = Htr.shape[0]
        k = int(n0 * args.augment_frac)
        for r in range(args.augment_rounds):
            sel = torch.randperm(n0, generator=g2)[:k]
            o = (torch.randn(k, 2, generator=g2) * args.augment_sigma).clamp(-args.augment_max, args.augment_max)
            Htr[sel, :, 0] += o[:, 0:1] / POS
            Htr[sel, :, 1] += o[:, 1:2] / POS
            Ctr[sel, :, 0] -= o[:, 0:1] / POS
            Ctr[sel, :, 1] -= o[:, 1:2] / POS
            Ytr[sel] -= o
        print(f"augmented {k * args.augment_rounds} samples (sigma={args.augment_sigma} max={args.augment_max})", flush=True)
    n = Htr.shape[0]
    print(f"samples: train={n} val={Hva.shape[0]} ({time.time()-t0:.1f}s build)", flush=True)

    dev = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
                       ("cpu" if args.device == "auto" else args.device))
    print(f"device={dev}", flush=True)
    net = AimNet().to(dev)
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
    lossf = torch.nn.SmoothL1Loss(beta=args.gamma or 1.0)
    b = args.batch
    if dev.type == "cuda":
        Htr, Ctr, Mtr, Ytr = Htr.to(dev), Ctr.to(dev), Mtr.to(dev), Ytr.to(dev)
        Hva, Cva, Mva, Yva = Hva.to(dev), Cva.to(dev), Mva.to(dev), Yva.to(dev)
    steps_per_epoch = (n + b - 1) // b
    total = steps_per_epoch * args.epochs
    sched = warmup_cosine(args.warmup, total, args.min_lr_factor)
    g = torch.Generator().manual_seed(args.seed)
    best = float("inf")
    best_sd = None
    gstep = 0
    for ep in range(1, args.epochs + 1):
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        te = time.time()
        for i in range(0, n, b):
            idx = perm[i:i + b]
            for gp in opt.param_groups:
                gp["lr"] = args.lr * sched(gstep)
            opt.zero_grad(set_to_none=True)
            loss = lossf(net(Htr[idx], Ctr[idx], Mtr[idx]), Ytr[idx])
            loss.backward()
            opt.step()
            gstep += 1
            tot += float(loss.detach()) * idx.shape[0]
        with torch.no_grad():
            net.eval()
            val = float(lossf(net(Hva, Cva, Mva), Yva))
            net.train()
        mark = ""
        if val < best:
            best = val
            best_sd = {k: v.detach().clone() for k, v in net.state_dict().items()}
            mark = " *"
        print(f"ep{ep:2d} train={tot/max(n,1):.5f} val={val:.5f} "
              f"({time.time()-te:.1f}s){mark}", flush=True)

    torch.save(best_sd, out / "aim.pt")
    src = Path(args.copy_from)
    for name in ("tap.pt", "read.pt"):
        if (src / name).exists() and not (out / name).exists():
            shutil.copy(src / name, out / name)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    (out / "_done.json").write_text(json.dumps({"status": "ok", "ts": stamp, "best": best}, indent=2))
    (out / "_DONE").write_text(f"ok {stamp} aim_val={best:.5f}\n")
    print(f"TRAINING DONE status=ok aim_val={best:.5f} -> {out/'aim.pt'}", flush=True)
    if args.notify_cmd:
        subprocess.run(args.notify_cmd, shell=True, check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
