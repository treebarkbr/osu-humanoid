"""Reward-driven search (human-anchored) evaluated on GitHub Actions.

We search a small, interpretable affine correction on top of the human-trained
nets, so the result stays close to the human policy (BC anchor = trust region):

    aim:  out'_i = g_i * out_i + o_i          i in {x, y}
    tap:  logit' = s * logit + t

Parameter vector x = [gx, gy, ox, oy, s, t], initialised at the identity
[1, 1, 0, 0, 1, 0]. CMA-ES drives it; the fitness is

    fit = sum_reward(maps) - lam * ||x - x0||^2

The best candidate is written to checkpoints_rl/{aim.pt,tap.pt} so the viewer's
RL button shows the trained result.

    .venv/bin/python scripts/rl_search.py --gens 12 --pop 8 --maps 8
"""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import scripts.rollout as RO  # noqa: E402
from pipeline.rl_params import NUM_PARAMS, STDS, X0, x_to_patch  # noqa: E402

CKPT_SEED = _ROOT / "checkpoints"
CKPT_RL = _ROOT / "checkpoints_rl"
REPO = "treebarkbr/osu-humanoid"
WORKFLOW = "eval.yml"

SUITE = ["924759.osu", "1021349.osu", "1092090.osu", "1093245.osu",
         "1093507.osu", "110175.osu", "1104798.osu", "1119026.osu"]


def _seed_nets():
    return RO._load(CKPT_SEED)


def write_rl_checkpoint(x: np.ndarray) -> None:
    CKPT_RL.mkdir(exist_ok=True)
    aim, tap = _seed_nets()
    cand = x_to_patch(x, aim, tap)
    for net, key, fname in ((aim, "aim_patch", "aim.pt"), (tap, "tap_patch", "tap.pt")):
        sd = net.state_dict()
        for k, v in cand[key].items():
            ref = sd[k]
            sd[k] = torch.tensor(v, dtype=ref.dtype).reshape(ref.shape)
        torch.save(sd, CKPT_RL / fname)


# ---------------------------------------------------------------------------
def gh(*args: str) -> str:
    return subprocess.run(["gh", *args], capture_output=True, text=True, check=True).stdout


def dispatch_and_collect(cands: list[dict], maps: list[str], shards: int, timeout: int = 1800) -> dict:
    payload = base64.b64encode(json.dumps(cands).encode()).decode()
    n_tasks = len(cands) * len(maps)
    n_shards = max(1, min(shards, n_tasks))
    shard_list = json.dumps(list(range(n_shards)))
    before = gh("run", "list", "--workflow", WORKFLOW, "--repo", REPO, "--limit", "1",
                "--json", "databaseId", "--jq", ".[0].databaseId").strip()
    gh("workflow", "run", WORKFLOW, "--repo", REPO,
       "-f", f"maps={','.join(maps)}", "-f", f"candidates_b64={payload}",
       "-f", f"shards={shard_list}")
    run_id = None
    t0 = time.time()
    while time.time() - t0 < 120:
        cur = gh("run", "list", "--workflow", WORKFLOW, "--repo", REPO, "--limit", "1",
                 "--json", "databaseId", "--jq", ".[0].databaseId").strip()
        if cur and cur != before:
            run_id = cur
            break
        time.sleep(3)
    if run_id is None:
        raise RuntimeError("could not find dispatched run id")
    print(f"  run {run_id} ({n_tasks} tasks / {n_shards} shards)", flush=True)

    t0 = time.time()
    while time.time() - t0 < timeout:
        j = json.loads(gh("run", "view", run_id, "--repo", REPO, "--json", "status,conclusion"))
        if j["status"] == "completed":
            print(f"  run {run_id} {j['conclusion']}", flush=True)
            break
        time.sleep(15)
    else:
        raise TimeoutError(f"run {run_id} did not finish")

    outdir = Path("/tmp/opencode/rl_dl")
    if outdir.exists():
        import shutil
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True)
    gh("run", "download", run_id, "--repo", REPO, "-D", str(outdir))

    by_cand: dict[str, list[dict]] = {}
    for f in outdir.rglob("*.json"):
        for r in json.loads(f.read_text()).get("rows", []):
            cid = r["id"].split("::", 1)[0]
            by_cand.setdefault(cid, []).append(r)
    return by_cand


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=12)
    ap.add_argument("--pop", type=int, default=8)
    ap.add_argument("--maps", type=int, default=8)
    ap.add_argument("--shards", type=int, default=16)
    ap.add_argument("--lam", type=float, default=150.0, help="trust-region weight (human anchor)")
    ap.add_argument("--log", default="/tmp/opencode/rl_search.log")
    args = ap.parse_args()

    maps = SUITE[: args.maps]
    import cma

    print(f"params dim {NUM_PARAMS}  maps {len(maps)}  pop {args.pop}", flush=True)
    es = cma.CMAEvolutionStrategy(X0.tolist(), 1.0,
                                  {"popsize": args.pop, "seed": 1, "verbose": -9,
                                   "maxiter": args.gens, "CMA_stds": STDS.tolist(),
                                   "bounds": [-30, 30]})

    logf = open(args.log, "a")

    def log(msg: str) -> None:
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    best_seen = -1e18
    for gen in range(args.gens):
        X = es.ask()
        cands = []
        for i, x in enumerate(X):
            cands.append({"id": f"g{gen}c{i}", "params": [float(v) for v in x]})
        by_cand = dispatch_and_collect(cands, maps, args.shards)

        fits, rewards = [], []
        for i, x in enumerate(X):
            rows = by_cand.get(f"g{gen}c{i}", [])
            R = sum(r.get("reward", 0.0) for r in rows if "error" not in r)
            accs = [r["acc"] for r in rows if "error" not in r]
            pen = args.lam * float(np.sum(((np.array(x) - X0) / STDS) ** 2))
            fits.append(R - pen)
            rewards.append(R)
            na = float(np.mean(accs)) if accs else 0.0
            log(f"gen {gen} c{i}: x={np.round(x,3).tolist()} R={R:9.0f} "
                f"acc={na*100:5.1f}% pen={pen:7.0f} fit={R-pen:9.0f}")

        bi = int(np.argmax(fits))
        if rewards[bi] > best_seen:
            best_seen = rewards[bi]
            write_rl_checkpoint(np.array(X[bi]))
            log(f"  >> new best R={rewards[bi]:.0f} x={np.round(X[bi],3).tolist()} "
                f"saved checkpoints_rl/")
        es.tell(X, fits)

    log(f"done. best reward {best_seen:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
