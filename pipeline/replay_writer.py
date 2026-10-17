"""Encode synthetic osu! replays to .osr (legacy format, stable/lazer readable).

We build a real ``osrparse.Replay`` and serialize it with osrparse's own
packer, so anything our reader (and the game) can parse round-trips exactly.
osrparse stores the replay hash as an osu-string (hex), not raw bytes.

Frames are given in absolute ms; the packer wants per-frame deltas, which
we compute here. The RNG-seed sentinel frame (w = -12345) is emitted by the
packer itself when ``rng_seed`` is set.
"""
from __future__ import annotations

import hashlib
import lzma
import struct
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from osrparse import (
    GameMode,
    Key,
    LifeBarState,
    Mod,
    Replay,
    ReplayEventOsu,
)
from osrparse.replay import _Packer

DEFAULT_GAME_VERSION = 20200205


def _compress_replay_data(data: bytes) -> bytes:
    """LZMA-compress replay frames exactly the way osu!stable/lazer write them.

    Python's ``lzma.compress(FORMAT_ALONE)`` writes an all-0xFF "unknown size"
    field in the 13-byte .lzma header. osu!'s custom LZMA decoder does not
    accept the sentinel (it reads it as a huge length and throws
    EndOfStreamException), so we patch in the real uncompressed size, matching
    what the game itself emits.
    """
    filters = [{"id": lzma.FILTER_LZMA1, "dict_size": 1 << 21, "mode": lzma.MODE_FAST}]
    compressed = lzma.compress(data, format=lzma.FORMAT_ALONE, filters=filters)
    return compressed[:5] + struct.pack("<Q", len(data)) + compressed[13:]


class _GamePacker(_Packer):
    """osrparse packer with a game-compatible LZMA header."""

    def pack_replay_data(self):
        data = ""
        for event in self.replay.replay_data:
            t = event.time_delta
            if isinstance(event, ReplayEventOsu):
                data += f"{t}|{event.x}|{event.y}|{event.keys.value},"
            else:
                data += f"{t}|{event.x}|{event.y}|{int(event.keys.value)},"
        if self.replay.rng_seed:
            data += f"-12345|0|0|{self.replay.rng_seed},"
        compressed = _compress_replay_data(data.encode("ascii"))
        return self.pack_int(len(compressed)) + compressed


@dataclass
class OutFrame:
    t: int      # absolute ms
    x: float
    y: float
    keys: int   # button bitfield (1=M1, 2=M2, 4=K1, 8=K2)


@dataclass
class OutReplay:
    beatmap_hash: str
    username: str = "osu-humanoid"
    mode: GameMode = GameMode.STD
    mods: int = 0
    game_version: int = DEFAULT_GAME_VERSION
    count300: int = 0
    count100: int = 0
    count50: int = 0
    count_geki: int = 0
    count_katu: int = 0
    count_miss: int = 0
    score: int = 0
    max_combo: int = 0
    perfect: bool = False
    life_bar_graph: list[tuple[int, float]] | None = None
    rng_seed: int | None = None
    replay_id: int = 0
    timestamp: datetime | None = None
    frames: list[OutFrame] = field(default_factory=list)


def _to_events(frames: list[OutFrame]) -> list[ReplayEventOsu]:
    events: list[ReplayEventOsu] = []
    prev = 0
    for f in frames:
        delta = f.t - prev
        prev = f.t
        events.append(ReplayEventOsu(delta, float(f.x), float(f.y), Key(int(f.keys))))
    return events


def write_replay(replay: OutReplay, path: str | Path) -> Path:
    path = Path(path)
    events = _to_events(replay.frames)
    frame_blob = ",".join(
        f"{e.time_delta}|{e.x}|{e.y}|{e.keys.value}," for e in events
    ).encode("ascii")
    replay_hash = hashlib.md5(frame_blob).hexdigest()

    life = None
    if replay.life_bar_graph is not None:
        life = [LifeBarState(t, life_val) for t, life_val in replay.life_bar_graph]

    rep = Replay(
        mode=replay.mode,
        game_version=replay.game_version,
        beatmap_hash=replay.beatmap_hash,
        username=replay.username,
        replay_hash=replay_hash,
        count_300=replay.count300,
        count_100=replay.count100,
        count_50=replay.count50,
        count_geki=replay.count_geki,
        count_katu=replay.count_katu,
        count_miss=replay.count_miss,
        score=replay.score,
        max_combo=replay.max_combo,
        perfect=replay.perfect,
        mods=Mod(replay.mods),
        life_bar_graph=life,
        timestamp=replay.timestamp or datetime.now(timezone.utc),
        replay_data=events,
        replay_id=replay.replay_id,
        rng_seed=replay.rng_seed,
    )
    data = _GamePacker(rep).pack()
    path.write_bytes(data)
    return path
