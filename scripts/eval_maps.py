"""Evaluate maps with the rollout + judge; print/write JSON.

Modes:
  * single suite:   --maps a.osu,b.osu  [--tap-patch p.json]
  * task file:      --tasks jobs/x.json [--shard i/n]

A task file is what the RL search writes; it lets one CI run evaluate many
candidate parameter sets across many maps, sharded over jobs::

    {
      "profile": "average", "seed": 0, "threshold": 0.5,
      "tasks": [
        {"id": "c0_galaxy", "map": "924759.osu", "tap_patch": {"net.4.bias": [0.1]}},
        ...
      ]
    }
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import scripts.rollout as RO  # noqa: E402
from pipeline.beatmap import parse_beatmap  # noqa: E402
from pipeline.humanize import from_profile  # noqa: E402
from pipeline.reward import od_for, radius_for, reward_map  # noqa: E402

BM_DIR = _ROOT / "data" / "beatmaps"


def apply_patch(net, patch: dict | None) -> None:
    if not patch:
        return
    sd = net.state_dict()
    for k, v in patch.items():
        ref = sd[k]
        sd[k] = torch.tensor(v, dtype=ref.dtype).reshape(ref.shape)
    net.load_state_dict(sd)


def apply_tap_patch(tap, patch: dict | None) -> None:
    apply_patch(tap, patch)


def _fresh_nets(ckpt: str):
    return RO._load(_ROOT / ckpt)


def eval_task(task: dict, *, ckpt: str, profile: str, seed: int, threshold: float) -> dict:
    bm_path = BM_DIR / task["map"]
    bm = parse_beatmap(bm_path)
    aim, tap = _fresh_nets(ckpt)
    if task.get("params") is not None:
        from pipeline.rl_params import x_to_patch
        p = x_to_patch(task["params"], aim, tap)
        apply_patch(aim, p["aim_patch"])
        apply_patch(tap, p["tap_patch"])
    else:
        apply_patch(tap, task.get("tap_patch"))
        apply_patch(aim, task.get("aim_patch"))
    ro = task.get("rollout") or {}
    kw = {"threshold": ro.get("threshold", threshold)}
    if "clip" in ro:
        kw["clip"] = ro["clip"]
    if "momentum" in ro:
        kw["momentum"] = ro["momentum"]
    rng = np.random.default_rng(seed)
    gen = RO.generate(bm, aim, tap, od=bm.od, ar=bm.ar, rng=rng,
                      params=from_profile(profile), **kw)
    rep = RO.judge_generated(bm, gen)
    c = rep.counts()
    reward, max_combo = reward_map(rep, radius=radius_for(bm), od=od_for(bm))
    return {
        "id": task.get("id", task["map"]),
        "map": task["map"],
        "title": bm.version,
        "objs": len(rep.results),
        "counts": c,
        "acc": rep.accuracy,
        "drops": rep.drops,
        "tick_misses": rep.tick_misses,
        "tick_hits": rep.tick_hits,
        "ur": rep.unstable_rate,
        "reward": reward,
        "max_combo": max_combo,
        "summary": rep.summary(),
    }


# Reward now lives in pipeline/reward.py (composite scheme B): per-object
# accuracy base + dense timing/tracking/aim/spin shaping + combo consistency.
# The old hand-tuned REWARD dict (drop/tick_miss/max_combo hacks) is gone.


def reward_from_report(rep, bm) -> tuple[float, int]:
    return reward_map(rep, radius=radius_for(bm), od=od_for(bm))


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {"maps": 0}
    tot = {k: sum(r["counts"].get(k, 0) for r in rows) for k in ("300", "100", "50", "miss")}
    hits = sum(tot.values())
    acc = (300 * tot["300"] + 100 * tot["100"] + 50 * tot["50"]) / (300 * hits) if hits else 1.0
    return {
        "maps": len(rows),
        "totals": tot,
        "acc": acc,
        "drops": sum(r["drops"] for r in rows),
        "tick_misses": sum(r["tick_misses"] for r in rows),
        "mean_acc": float(np.mean([r["acc"] for r in rows])),
        "reward": sum(r["reward"] for r in rows),
        "mean_reward": float(np.mean([r["reward"] for r in rows])),
        "per_map": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", default="924759.osu", help="comma list of .osu filenames")
    ap.add_argument("--all", action="store_true", help="use every map in data/beatmaps")
    ap.add_argument("--tasks", default=None, help="path to a task JSON file")
    ap.add_argument("--candidates-b64", default=None,
                    help="base64 JSON list of {id, tap_patch}; crossed with --maps")
    ap.add_argument("--shard", default=None, help="i/n -> evaluate every n-th task starting at i")
    ap.add_argument("--ckpt", default="checkpoints")
    ap.add_argument("--profile", default="average")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--tap-patch", default=None, help="path to JSON patch for tap params")
    ap.add_argument("--out", default=None, help="write JSON result here")
    args = ap.parse_args()

    profile, seed, threshold, ckpt = args.profile, args.seed, args.threshold, args.ckpt

    if args.candidates_b64:
        import base64
        cands = json.loads(base64.b64decode(args.candidates_b64))
        files = [f.strip() for f in args.maps.split(",") if f.strip()]
        tasks = [{"id": f"{c['id']}::{m}", "map": m,
                  **{k: v for k, v in c.items() if k != "id"}}
                 for c in cands for m in files]
    elif args.tasks:
        cfg = json.loads(Path(args.tasks).read_text())
        profile = cfg.get("profile", profile)
        seed = cfg.get("seed", seed)
        threshold = cfg.get("threshold", threshold)
        ckpt = cfg.get("ckpt", ckpt)
        tasks = cfg["tasks"]
    else:
        if args.all:
            files = sorted(p.name for p in BM_DIR.glob("*.osu"))
        else:
            files = [f.strip() for f in args.maps.split(",") if f.strip()]
        patch = json.loads(Path(args.tap_patch).read_text()) if args.tap_patch else None
        tasks = [{"id": f, "map": f, "tap_patch": patch} for f in files]

    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        tasks = tasks[i::n]

    rows = []
    for t in tasks:
        try:
            r = eval_task(t, ckpt=ckpt, profile=profile, seed=seed, threshold=threshold)
        except Exception as e:  # noqa: BLE001
            r = {"id": t.get("id", t["map"]), "map": t["map"], "error": f"{type(e).__name__}: {e}"}
        rows.append(r)
        print(json.dumps(r), flush=True)

    good = [r for r in rows if "error" not in r]
    agg = aggregate(good)
    agg["errors"] = [r for r in rows if "error" in r]
    print("AGG " + json.dumps({k: v for k, v in agg.items() if k != "per_map"}), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps({"rows": rows, "agg": agg}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
