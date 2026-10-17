#!/usr/bin/env python3
"""Fetch ranked osu!std difficulties (raw .osu files) from the hinai mirror.

No auth required. Polite rate limit. Writes data/beatmaps/{beatmap_id}.osu
and maintains data/beatmaps/index.csv.

Usage:
  python scripts/fetch_maps.py --min-stars 4.0 --max-stars 6.0 --count 30
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import requests

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "data" / "beatmaps"
INDEX = OUT / "index.csv"

UA = "osu-humanoid/0.1 (research project; local ML pipeline) by nocone"
SEARCH = "https://mirror.hinamizawa.ai/api/v1/hinai/search"
RAW_OSU = "https://mirror.hinamizawa.ai/api/osu/{bid}"

SESSION = requests.Session()
SESSION.headers["User-Agent"] = UA


def search_page(min_stars: float, max_stars: float, amount: int, offset: int) -> list[dict]:
    r = SESSION.get(
        SEARCH,
        params={
            "status": 1,          # ranked
            "mode": 0,            # osu!standard
            "min_stars": min_stars,
            "max_stars": max_stars,
            "amount": amount,
            "offset": offset,
            "sort": "plays_desc",
            "explicit": "hide",
        },
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        raise RuntimeError(f"unexpected search response: {data!r:.200}")
    return data


def load_index() -> dict[int, dict]:
    rows: dict[int, dict] = {}
    if INDEX.exists():
        with INDEX.open() as f:
            for row in csv.DictReader(f):
                rows[int(row["beatmap_id"])] = row
    return rows


def save_index_row(row: dict, existing: dict[int, dict]) -> None:
    existing[int(row["beatmap_id"])] = row
    with INDEX.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "beatmap_id", "set_id", "stars", "version",
                "artist", "title", "creator", "cs", "ar", "od", "path",
            ],
        )
        w.writeheader()
        w.writerows(existing.values())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-stars", type=float, default=4.0)
    ap.add_argument("--max-stars", type=float, default=6.0)
    ap.add_argument("--count", type=int, default=20, help="number of difficulties to download")
    ap.add_argument("--delay", type=float, default=0.4, help="seconds between downloads")
    ap.add_argument("--max-per-set", type=int, default=1)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    have = load_index()

    # collect candidate difficulties from search pages
    candidates: list[dict] = []
    seen_sets: set[int] = set()
    offset = 0
    while len(candidates) < args.count * 3:
        page = search_page(args.min_stars, args.max_stars, 100, offset)
        if not page:
            break
        for s in page:
            sid = s.get("SetID")
            if sid in seen_sets:
                continue
            seen_sets.add(sid)
            for d in s.get("ChildrenBeatmaps", []):
                if d.get("Mode", 0) != 0:
                    continue
                stars = d.get("DifficultyRating", 0)
                if not (args.min_stars <= stars <= args.max_stars):
                    continue
                if d.get("BeatmapID", 0) in have:
                    continue
                candidates.append(
                    {
                        "beatmap_id": d["BeatmapID"],
                        "set_id": sid,
                        "stars": stars,
                        "version": d.get("DiffName", ""),
                        "artist": s.get("Artist", ""),
                        "title": s.get("Title", ""),
                        "creator": s.get("Creator", ""),
                        "ar": d.get("AR", ""),
                        "cs": d.get("CS", ""),
                        "od": d.get("OD", ""),
                    }
                )
        offset += 100
        time.sleep(0.2)

    # prefer a spread of star ratings
    candidates.sort(key=lambda c: c["stars"])
    if candidates:
        step = max(1, len(candidates) // max(1, args.count))
        picked = candidates[::step][: args.count]
    else:
        picked = []

    if not picked:
        print("nothing to download (all already indexed?)")
        return 0

    print(f"downloading {len(picked)} difficulties "
          f"({args.min_stars}-{args.max_stars}★)...")
    for i, c in enumerate(picked, 1):
        dest = OUT / f"{c['beatmap_id']}.osu"
        if dest.exists():
            print(f"  [{i}/{len(picked)}] {c['beatmap_id']} exists, indexing")
        else:
            try:
                r = SESSION.get(RAW_OSU.format(bid=c["beatmap_id"]), timeout=30)
                r.raise_for_status()
                text = r.text
                if "osu file format" not in text[:100]:
                    print(f"  [{i}/{len(picked)}] {c['beatmap_id']} BAD RESPONSE, skip")
                    time.sleep(args.delay)
                    continue
                dest.write_text(text, encoding="utf-8")
                print(f"  [{i}/{len(picked)}] {c['beatmap_id']} "
                      f"{c['artist']} - {c['title']} [{c['version']}] {c['stars']}★")
            except Exception as e:
                print(f"  [{i}/{len(picked)}] {c['beatmap_id']} FAILED: {e}")
                time.sleep(args.delay)
                continue
        c["path"] = str(dest)
        save_index_row(c, have)
        time.sleep(args.delay)

    print(f"done. index: {INDEX} ({len(have)} total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
