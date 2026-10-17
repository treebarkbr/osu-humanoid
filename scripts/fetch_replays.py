#!/usr/bin/env python3
"""Download leaderboard replays (+ their .osu files) via the osu!api v2.

Requires a valid token: run ``scripts/get_osu_token.py`` first.

For each beatmap it fetches the global leaderboard (top scores), then downloads
each replay with ``GET /scores/osu/{score}/download``. The matching ``.osu`` file
is pulled from the hinai mirror, and the replay's embedded beatmap hash is
verified against it.

Usage:
  python scripts/fetch_replays.py --from-index --limit 20
  python scripts/fetch_replays.py --beats 126645,1932649 --limit 10
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import osu_api  # noqa: E402
from pipeline.replay import load_replay  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
MAPS = BASE / "data" / "beatmaps"
REPLAYS = BASE / "data" / "replays"
INDEX = REPLAYS / "index.csv"

RAW_OSU = "https://mirror.hinamizawa.ai/api/osu/{bid}"
UA = osu_api.UA

MIRROR = requests.Session()
MIRROR.headers["User-Agent"] = UA

INDEX_FIELDS = [
    "beatmap_id", "score_id", "user_id", "username", "mods", "score", "acc",
    "count300", "count100", "count50", "miss", "max_combo", "replay_path", "map_path",
]


def load_index() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if INDEX.exists():
        with INDEX.open() as f:
            for row in csv.DictReader(f):
                rows[row["score_id"]] = row
    return rows


def save_index(rows: dict[str, dict]) -> None:
    REPLAYS.mkdir(parents=True, exist_ok=True)
    with INDEX.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        w.writeheader()
        w.writerows(rows.values())


def ensure_map(beatmap_id: int) -> Path | None:
    dest = MAPS / f"{beatmap_id}.osu"
    if dest.exists():
        return dest
    try:
        r = MIRROR.get(RAW_OSU.format(bid=beatmap_id), timeout=30)
        r.raise_for_status()
        if "osu file format" not in r.text[:100]:
            return None
        dest.write_text(r.text, encoding="utf-8")
        return dest
    except requests.RequestException as e:
        print(f"    map {beatmap_id} download failed: {e}")
        return None


def map_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def mods_str(score: dict) -> str:
    return "".join(m["acronym"] if isinstance(m, dict) else str(m)
                   for m in score.get("mods", []))


def fetch_beatmap_scores(beatmap_id: int) -> list[dict]:
    data = osu_api.api_get(f"/beatmaps/{beatmap_id}/scores", params={"mode": "osu"})
    if data is None:
        return []
    if isinstance(data, dict):
        return data.get("scores", []) or []
    return data or []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-index", action="store_true",
                    help="use every beatmap id in data/beatmaps/index.csv")
    ap.add_argument("--beats", help="comma-separated beatmap ids")
    ap.add_argument("--limit", type=int, default=15, help="max scores per beatmap")
    ap.add_argument("--delay", type=float, default=1.1, help="seconds between API calls")
    args = ap.parse_args()

    if args.beats:
        beat_ids = [int(x) for x in args.beats.split(",") if x.strip()]
    elif args.from_index:
        rows = list(csv.DictReader((MAPS / "index.csv").open()))
        beat_ids = [int(r["beatmap_id"]) for r in rows]
    else:
        ap.error("pass --from-index or --beats")

    REPLAYS.mkdir(parents=True, exist_ok=True)
    index = load_index()
    print(f"fetching replays for {len(beat_ids)} beatmaps (limit {args.limit}/map)")

    for bi, bid in enumerate(beat_ids, 1):
        scores = fetch_beatmap_scores(bid)
        time.sleep(args.delay)
        if not scores:
            print(f"  [{bi}/{len(beat_ids)}] {bid}: no scores")
            continue

        map_path = ensure_map(bid)
        want_md5 = map_md5(map_path) if map_path else None

        got = 0
        for score in scores[: args.limit]:
            sid = score.get("id")
            if sid is None or str(sid) in index:
                continue
            try:
                blob = osu_api.api_get(f"/scores/osu/{sid}/download", raw=True)
            except Exception as e:
                print(f"    score {sid}: download failed: {e}")
                time.sleep(args.delay)
                continue
            time.sleep(args.delay)

            if not blob or len(blob) < 64:
                print(f"    score {sid}: empty replay")
                continue

            dest_dir = REPLAYS / str(bid)
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / f"{sid}.osr"
            dest.write_bytes(blob)
            try:
                rep = load_replay(dest)
            except Exception as e:
                print(f"    score {sid}: unparseable replay ({e})")
                dest.unlink(missing_ok=True)
                continue

            ok = "ok" if (want_md5 and rep.beatmap_hash == want_md5) else "MISMATCH"
            stats = score.get("statistics", {}) or {}
            index[str(sid)] = {
                "beatmap_id": bid,
                "score_id": sid,
                "user_id": score.get("user_id", ""),
                "username": score.get("user", {}).get("username", "") if isinstance(score.get("user"), dict) else "",
                "mods": mods_str(score),
                "score": score.get("score", ""),
                "acc": score.get("accuracy", ""),
                "count300": stats.get("count_300", ""),
                "count100": stats.get("count_100", ""),
                "count50": stats.get("count_50", ""),
                "miss": stats.get("count_miss", ""),
                "max_combo": score.get("max_combo", ""),
                "replay_path": str(dest),
                "map_path": str(map_path) if map_path else "",
            }
            got += 1
            print(f"    score {sid} by {index[str(sid)]['username']} "
                  f"[{index[str(sid)]['mods']}] hash={ok}")

        save_index(index)
        print(f"  [{bi}/{len(beat_ids)}] {bid}: +{got} replays (total {len(index)})")

    print(f"done. {len(index)} replays -> {INDEX}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
