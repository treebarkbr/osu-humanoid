"""Modal fan-out for osu-humanoid rollouts.

Runs ``scripts.rollout.generate`` + the headless judge in the cloud so we can
evaluate many (candidate, map) pairs in parallel for free.

Smoke test:
    .venv/bin/python -m modal run modal_rollout.py --maps 924759.osu
Fan-out (returns full summaries):
    .venv/bin/python -m modal run modal_rollout.py --maps 924759.osu 1021349.osu --fanout
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

PROJ = "/root/proj"

app = modal.App("osu-humanoid")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "torch", extra_index_url="https://download.pytorch.org/whl/cpu")
    .add_local_dir("models", f"{PROJ}/models")
    .add_local_dir("pipeline", f"{PROJ}/pipeline")
    .add_local_dir("scripts", f"{PROJ}/scripts")
    .add_local_dir("checkpoints", f"{PROJ}/checkpoints")
    .add_local_dir("data/beatmaps", f"{PROJ}/data/beatmaps")
    .add_local_dir("data/geometry", f"{PROJ}/data/geometry")
)


@app.function(image=image, cpu=4, memory=4096, timeout=900)
def rollout_one(
    map_file: str,
    profile: str = "average",
    seed: int = 0,
    ckpt: str = "checkpoints",
    threshold: float = 0.5,
    tap_patch: dict | None = None,
) -> dict:
    import sys

    if PROJ not in sys.path:
        sys.path.insert(0, PROJ)

    import numpy as np
    import torch

    import scripts.rollout as RO
    from pipeline.beatmap import parse_beatmap
    from pipeline.humanize import from_profile

    bm = parse_beatmap(Path(PROJ) / "data" / "beatmaps" / map_file)
    aim, tap = RO._load(Path(PROJ) / ckpt)

    if tap_patch is not None:
        with torch.no_grad():
            for name, vals in tap_patch.items():
                p = tap
                parts = name.split(".")
                for pp in parts[:-1]:
                    p = getattr(p, pp) if not pp.isdigit() else p[int(pp)]
                setattr(p, parts[-1], torch.tensor(vals, dtype=getattr(p, parts[-1]).dtype))

    params = from_profile(profile)
    rng = np.random.default_rng(seed)
    gen = RO.generate(bm, aim, tap, od=bm.od, ar=bm.ar, rng=rng,
                      params=params, threshold=threshold)
    rep = RO.judge_generated(bm, gen)
    c = rep.counts()
    return {
        "map": map_file,
        "objs": len(rep.results),
        "counts": c,
        "acc": rep.accuracy,
        "drops": rep.drops,
        "tick_misses": rep.tick_misses,
        "tick_hits": rep.tick_hits,
        "ur": rep.unstable_rate,
        "summary": rep.summary(),
    }


@app.local_entrypoint()
def main(maps: str = "924759.osu", fanout: bool = False, profile: str = "average", seed: int = 0):
    files = maps.split(",")
    if fanout:
        results = list(rollout_one.map(files, kwargs={"profile": profile, "seed": seed}))
        for r in results:
            print(json.dumps(r))
    else:
        r = rollout_one.remote(files[0], profile=profile, seed=seed)
        print(json.dumps(r, indent=2))
