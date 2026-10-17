"""Reward-driven search (human-anchored) evaluated on GitHub Actions.

We search a small, interpretable affine correction on top of the human-trained
nets, so the result stays close to the human policy (BC anchor = trust region):

    aim:  out'_i = g_i * out_i + o_i          i in {x, y}
    tap:  logit' = s * logit + t

Parameter vector x = [gx, gy, ox, oy, s, t], initialised at the identity
[1, 1, 0, 0, 1, 0]. A robust compass + random search (one dispatch per
generation, only moves on improvement) maximises

    fit = mean_map(reward)

where `reward` is the composite judge reward (pipeline/reward.py).

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
    ap.add_argument("--log", default="/tmp/opencode/rl_search.log")
    args = ap.parse_args()

    maps = SUITE[: args.maps]
    dim = NUM_PARAMS

    logf = open(args.log, "a")

    def log(msg: str) -> None:
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    # Parameter box: keeps the search physically sane (gains ~1, offsets
    # px/frame, tap bias in logits). CMA-ES drifted off this manifold, so we
    # use a robust compass + random search that can only move on improvement.
    lo = np.array([0.6, 0.6, -0.3, -0.3, 0.5, -3.0])
    hi = np.array([1.4, 1.4, 0.3, 0.3, 1.5, 3.0])

    def clamp(v):
        return np.clip(v, lo, hi)

    def evaluate(param_list):
        cands = [{"id": cid, "params": [float(v) for v in vv]} for cid, vv in param_list]
        by = dispatch_and_collect(cands, maps, args.shards)
        out = {}
        for cid, _ in param_list:
            rows = [r for r in by.get(cid, []) if "error" not in r]
            R = float(np.mean([r["reward"] for r in rows])) if rows else -1e18
            acc = float(np.mean([r["acc"] for r in rows])) if rows else 0.0
            out[cid] = (R, acc)
        return out

    # Seed baseline (identity == human nets): start the incumbent here and
    # write it into checkpoints_rl so we can never save worse-than-human.
    x = X0.copy()
    R_cur, acc_cur = evaluate([("seed", x)])["seed"]
    write_rl_checkpoint(x)
    log(f"seed x0: R={R_cur:.4f} acc={acc_cur*100:.2f}%")
    best_R, best_x, best_acc = R_cur, x.copy(), acc_cur

    step = STDS.copy()
    rng = np.random.default_rng(1)
    for gen in range(args.gens):
        probes = []
        for i in range(dim):
            for sgn in (1.0, -1.0):
                y = x.copy()
                y[i] += sgn * step[i]
                probes.append((f"g{gen}_a{i}{'p' if sgn > 0 else 'n'}", clamp(y)))
        for i in range(args.pop):
            probes.append((f"g{gen}_r{i}", clamp(x + rng.standard_normal(dim) * step)))

        res = evaluate(probes)
        bx, bR, bA = None, -1e18, 0.0
        for cid, y in probes:
            R, acc = res[cid]
            log(f"gen {gen} {cid}: R={R:.4f} acc={acc*100:5.1f}%")
            if R > bR:
                bx, bR, bA = y, R, acc
        if bx is not None and bR > R_cur + 1e-4:
            x, R_cur, acc_cur = bx.copy(), bR, bA
            if R_cur > best_R:
                best_R, best_x, best_acc = R_cur, x.copy(), acc_cur
                write_rl_checkpoint(x)
                log(f"  >> new best R={best_R:.4f} acc={best_acc*100:.2f}% "
                    f"x={np.round(x,3).tolist()}")
        else:
            step = step * 0.5
            log(f"  stall -> step*0.5 {np.round(step,4).tolist()}")

    log(f"done. best mean reward {best_R:.4f} acc={best_acc*100:.2f}% "
        f"x={np.round(best_x,3).tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
