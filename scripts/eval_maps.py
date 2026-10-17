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

BM_DIR = _ROOT / "data" / "beatmaps"


def apply_tap_patch(tap, patch: dict | None) -> None:
    if not patch:
        return
    sd = tap.state_dict()
    for k, v in patch.items():
        ref = sd[k]
        sd[k] = torch.tensor(v, dtype=ref.dtype).reshape(ref.shape)
    tap.load_state_dict(sd)


def _fresh_nets(ckpt: str):
    return RO._load(_ROOT / ckpt)


def eval_task(task: dict, *, ckpt: str, profile: str, seed: int, threshold: float) -> dict:
    bm_path = BM_DIR / task["map"]
    bm = parse_beatmap(bm_path)
    aim, tap = _fresh_nets(ckpt)
    apply_tap_patch(tap, task.get("tap_patch"))
    rng = np.random.default_rng(seed)
    gen = RO.generate(bm, aim, tap, od=bm.od, ar=bm.ar, rng=rng,
                      params=from_profile(profile), threshold=threshold)
    rep = RO.judge_generated(bm, gen)
    c = rep.counts()
    reward, max_combo = reward_from_report(rep)
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


# Reward weights (user spec). Hits -> +20/+6/+2, miss -> -40, slider break ->
# -60, tick miss -> -15, per-combo-step +0.2, new-max-combo +30, perfect hit +2.
REWARD = {
    "h300": 20.0, "h100": 6.0, "h50": 2.0, "miss": -40.0,
    "drop": -60.0, "tick_miss": -15.0, "tick_hit": 0.5,
    "combo_step": 0.2, "max_combo": 30.0, "perfect": 2.0, "perfect_ms": 8.0,
    "spin_300": 25.0, "spin_100": 8.0, "spin_50": 3.0, "spin_miss": -40.0,
}


def reward_from_report(rep) -> tuple[float, int]:
    R = 0.0
    combo = 0
    max_combo = 0
    for r in rep.results:
        if r.is_spinner:
            R += {"300": REWARD["spin_300"], "100": REWARD["spin_100"],
                  "50": REWARD["spin_50"]}.get(r.kind, REWARD["spin_miss"])
            continue
        if r.kind == "miss":
            R += REWARD["miss"]
            combo = 0
        else:
            R += REWARD[f"h{r.kind}"]
            combo += 1
            R += REWARD["combo_step"]
            if combo > max_combo:
                max_combo = combo
                R += REWARD["max_combo"]
            if r.kind == "300" and r.hit_error is not None and abs(r.hit_error) < REWARD["perfect_ms"]:
                R += REWARD["perfect"]
        if r.is_slider:
            R += REWARD["drop"] * r.dropped
            R += REWARD["tick_miss"] * r.tick_misses
            R += REWARD["tick_hit"] * r.tick_hits
    return R, max_combo


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
        tasks = [{"id": f"{c['id']}::{m}", "map": m, "tap_patch": c.get("tap_patch")}
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
