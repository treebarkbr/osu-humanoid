"""Evaluate trained heads on the held-out (group-split) validation set.

    .venv/bin/python scripts/eval_models.py --stride 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import AimNet, ReadingNet, TappingNet  # noqa: E402
from pipeline.torch_data import build_samples, split  # noqa: E402

POS = 512.0


def pr_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-scores)
    lab = labels[order].astype(np.float64)
    tp = np.cumsum(lab)
    fp = np.cumsum(1.0 - lab)
    prec = tp / np.maximum(tp + fp, 1e-9)
    rec = tp / max(lab.sum(), 1.0)
    prec = np.concatenate(([1.0], prec))
    rec = np.concatenate(([0.0], rec))
    return float(np.trapezoid(prec, rec))


def best_f1(scores: np.ndarray, labels: np.ndarray):
    best = (0.0, 0.0)
    for th in np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 200))):
        pred = scores >= th
        tp = float(np.sum(pred & (labels > 0)))
        fp = float(np.sum(pred & (labels == 0)))
        fn = float(np.sum(~pred & (labels > 0)))
        f1 = 2 * tp / max(2 * tp + fp + fn, 1e-9)
        if f1 > best[0]:
            best = (f1, float(th))
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default="data/datasets")
    ap.add_argument("--ckpt", default="checkpoints")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--limit-shards", type=int, default=0)
    args = ap.parse_args()

    data = build_samples(args.dataset_dir, max_shards=args.limit_shards, stride=args.stride)
    _tr, va = split(data, args.val_frac)
    ck = Path(args.ckpt)

    with torch.no_grad():
        aim = AimNet(); aim.load_state_dict(torch.load(ck / "aim.pt")); aim.eval()
        pa = aim(va["hist"], va["ctx"], va["meta"]).numpy()
        ea = np.linalg.norm(pa - va["aim"].numpy(), axis=1)

        rd = ReadingNet(); rd.load_state_dict(torch.load(ck / "read.pt")); rd.eval()
        pr = rd(va["hist"], va["ctx"], va["meta"]).numpy()
        er = np.linalg.norm(pr - va["read"].numpy(), axis=1) * POS

        tp = TappingNet(); tp.load_state_dict(torch.load(ck / "tap.pt")); tp.eval()
        logit = tp(va["hist"], va["ctx"], va["meta"]).numpy().reshape(-1)
        prob = 1.0 / (1.0 + np.exp(-logit))
        lab = va["tap"].numpy().reshape(-1)

    print(f"val samples: {len(ea)}")
    print(f"[aim ] next-frame delta err (px/frame): mean={ea.mean():.3f}px "
          f"median={np.median(ea):.3f}px  p90={np.percentile(ea, 90):.3f}px")
    print(f"[read] cursor@hit-time err       : mean={er.mean():.2f}px "
          f"median={np.median(er):.2f}px  p90={np.percentile(er, 90):.2f}px")
    cur = va["hist"].numpy()[:, -1, :2]  # current cursor (normalised)
    base = np.linalg.norm(va["read"].numpy() - cur, axis=1) * POS  # "cursor stays put"
    print(f"[read] baseline (cursor-stays) err: mean={base.mean():.2f}px")
    pos_rate = float(lab.mean())
    print(f"[tap ] press rate={pos_rate * 100:.2f}%  PR-AUC={pr_auc(prob, lab):.4f} "
          f"best_f1={best_f1(prob, lab)[0]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
