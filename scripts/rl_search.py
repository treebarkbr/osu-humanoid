"""Reward-driven search over the tap head, evaluated on GitHub Actions.

Outer loop (this script, local):
  * CMA-ES over the tap net's last layer (weight+bias) plus a couple of scalar
    press knobs, initialised at the seed (human-trained) checkpoint.
  * each generation: write the population as candidates, dispatch the `eval`
    workflow on a suite of maps, collect per-map rewards, and compute
        fitness = sum(reward) - lam * ||x - x0||^2        (trust region = human anchor)
  * the best candidate is written back to checkpoints_rl/tap.pt.

Usage:
    .venv/bin/python scripts/rl_search.py --gens 10 --pop 8 --maps 8
"""
from __future__ import annotations

import argparse
import base64
import json
import math
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
from pipeline.torch_data import W, K  # noqa: E402
import cma  # noqa: E402

CKPT_SEED = _ROOT / "checkpoints"
CKPT_RL = _ROOT / "checkpoints_rl"
REPO = "treebarkbr/osu-humanoid"
WORKFLOW = "eval.yml"

SUITE = ["924759.osu", "1021349.osu", "1092090.osu", "1093245.osu",
         "1093507.osu", "110175.osu", "1104798.osu", "1119026.osu"]


# ---------------------------------------------------------------------------
def load_seed_vec(ckpt: Path) -> np.ndarray:
    _, tap = RO._load(ckpt)
    sd = tap.state_dict()
    return np.concatenate([sd["net.4.weight"].reshape(-1).numpy(),
                           sd["net.4.bias"].reshape(-1).numpy()]).astype(np.float64)


def vec_to_patch(x: np.ndarray) -> dict:
    nw = W * 5 + K * 4 + 4  # 96 for the act-enabled net
    # infer hidden width from the seed weight length: len = hidden + 1 (bias)
    hidden = x.shape[0] - 1
    w = x[:hidden].reshape(1, hidden).tolist()
    b = [float(x[hidden])]
    return {"net.4.weight": w, "net.4.bias": b}


def write_rl_checkpoint(x: np.ndarray) -> None:
    CKPT_RL.mkdir(exist_ok=True)
    aim_src = CKPT_SEED / "aim.pt"
    (CKPT_RL / "aim.pt").write_bytes(aim_src.read_bytes())
    _, tap = RO._load(CKPT_SEED)
    sd = tap.state_dict()
    patch = vec_to_patch(x)
    sd["net.4.weight"] = torch.tensor(patch["net.4.weight"], dtype=sd["net.4.weight"].dtype).reshape(sd["net.4.weight"].shape)
    sd["net.4.bias"] = torch.tensor(patch["net.4.bias"], dtype=sd["net.4.bias"].dtype).reshape(sd["net.4.bias"].shape)
    torch.save(sd, CKPT_RL / "tap.pt")


# ---------------------------------------------------------------------------
def gh(*args: str) -> str:
    return subprocess.run(["gh", *args], capture_output=True, text=True, check=True).stdout


def dispatch_and_collect(cands: list[dict], maps: list[str], shards: int, timeout: int = 1200) -> dict:
    payload = base64.b64encode(json.dumps(cands).encode()).decode()
    n_tasks = len(cands) * len(maps)
    n_shards = max(1, min(shards, n_tasks))
    shard_list = json.dumps(list(range(n_shards)))
    before = gh("run", "list", "--workflow", WORKFLOW, "--repo", REPO, "--limit", "1",
                "--json", "databaseId", "--jq", ".[0].databaseId").strip()
    gh("workflow", "run", WORKFLOW, "--repo", REPO,
       "-f", f"maps={','.join(maps)}", "-f", f"candidates_b64={payload}",
       "-f", f"shards={shard_list}")
    # find the new run id
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
        st = gh("run", "view", run_id, "--repo", REPO, "--json", "status,conclusion")
        j = json.loads(st)
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
        d = json.loads(f.read_text())
        for r in d.get("rows", []):
            cid, _, mp = r["id"].partition("::")
            by_cand.setdefault(cid, []).append(r)
    return by_cand


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=10)
    ap.add_argument("--pop", type=int, default=8)
    ap.add_argument("--maps", type=int, default=8, help="how many suite maps to use")
    ap.add_argument("--shards", type=int, default=16)
    ap.add_argument("--sigma0", type=float, default=0.25)
    ap.add_argument("--lam", type=float, default=0.5, help="trust-region weight")
    ap.add_argument("--log", default="/tmp/opencode/rl_search.log")
    args = ap.parse_args()

    maps = SUITE[: args.maps]
    x0 = load_seed_vec(CKPT_SEED)
    print(f"seed vec dim {x0.shape[0]}  maps {len(maps)}  pop {args.pop}", flush=True)

    es = cma.CMAEvolutionStrategy(x0.tolist(), args.sigma0,
                                  {"popsize": args.pop, "seed": 0,
                                   "verbose": -9, "maxiter": args.gens})
    logf = open(args.log, "a")

    def log(msg: str) -> None:
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    best_seen = -1e18
    gen = 0
    while not es.stop() and gen < args.gens:
        X = es.ask()
        cands = [{"id": f"g{gen}c{i}", "tap_patch": vec_to_patch(np.array(x))}
                 for i, x in enumerate(X)]
        by_cand = dispatch_and_collect(cands, maps, args.shards)

        fits = []
        rewards = []
        for i, x in enumerate(X):
            cid = f"g{gen}c{i}"
            rows = by_cand.get(cid, [])
            R = sum(r.get("reward", 0.0) for r in rows if "error" not in r)
            accs = [r["acc"] for r in rows if "error" not in r]
            pen = args.lam * float(np.sum((np.array(x) - x0) ** 2))
            fit = R - pen
            fits.append(fit)
            rewards.append(R)
            na = float(np.mean(accs)) if accs else 0.0
            log(f"gen {gen} c{i}: R={R:9.0f} acc={na*100:5.1f}% pen={pen:8.0f} fit={fit:9.0f}")

        best_i = int(np.argmax(fits))
        if rewards[best_i] > best_seen:
            best_seen = rewards[best_i]
            write_rl_checkpoint(np.array(X[best_i]))
            log(f"  >> new best R={rewards[best_i]:.0f} saved checkpoints_rl/tap.pt")

        es.tell(X, fits)
        gen += 1

    log(f"done. best reward {best_seen:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
