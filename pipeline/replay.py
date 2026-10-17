"""Replay (.osr) loading via osrparse → flat numpy-friendly frames."""
from __future__ import annotations

import lzma
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from osrparse import Replay, parse_replay_data

PLAYFIELD_W, PLAYFIELD_H = 512.0, 384.0


@dataclass
class ReplayFrames:
    times: np.ndarray     # int64, absolute ms
    x: np.ndarray         # float32, osu!pixels (0..512 / 0..384)
    y: np.ndarray
    keys: np.ndarray      # uint8, bit0 = M1/K1, bit1 = M2/K2
    mods: tuple
    count300: int
    count100: int
    count50: int
    count_miss: int
    max_combo: int
    beatmap_hash: str
    username: str
    game_version: int = 0


def _read_uleb(b: bytes, i: int) -> tuple[int, int]:
    val = 0
    shift = 0
    while True:
        c = b[i]
        i += 1
        val |= (c & 0x7F) << shift
        if not (c & 0x80):
            break
        shift += 7
    return val, i


def _read_osu_str(b: bytes, i: int) -> tuple[str, int]:
    if b[i] == 0x00:
        return "", i + 1
    i += 1  # 0x0b marker
    ln, i = _read_uleb(b, i)
    return b[i:i + ln].decode("utf-8", "replace"), i + ln


def _raw_replay_string(path: Path) -> str:
    """Decompress the replay-data blob straight out of the .osr.

    osrparse parses this for us too, but it *drops* the leading sentinel
    frames whose second field carries the replay's time-base offset.
    """
    b = Path(path).read_bytes()
    i = 1 + 4  # gamemode (1) + version (4)
    _, i = _read_osu_str(b, i)  # beatmap hash
    _, i = _read_osu_str(b, i)  # player name
    _, i = _read_osu_str(b, i)  # replay md5
    i += 2 * 6  # count300/100/50/geki/katu/miss
    i += 4  # score
    i += 2  # max combo
    i += 1  # perfect
    i += 4  # mods
    _, i = _read_osu_str(b, i)  # life bar
    i += 8  # timestamp
    ln = struct.unpack_from("<i", b, i)[0]
    i += 4
    blob = b[i:i + ln]
    return lzma.decompress(blob).decode("utf-8", "replace")


def _time_base_offset(path: Path) -> int:
    """Offset to add to every frame time (second sentinel's time field).

    osu!stable writes ``0|256|-500|0,<S>|256|-500|0,<delta>|...``. The first
    sentinel is the RNG seed; the second holds ``S``, the absolute time the
    frame deltas are measured from. Frames with S != 0 (common for lazer and
    for replays that started before the map) come out ~3s wrong otherwise.
    """
    try:
        raw = _raw_replay_string(path)
    except Exception:
        return 0
    toks = [t for t in raw.replace("\n", ",").split(",") if t]
    if len(toks) >= 2:
        f0 = toks[0].split("|")
        f1 = toks[1].split("|")
        if (len(f0) == 4 and len(f1) == 4
                and f0[1] == "256" and f0[2] == "-500"
                and f1[1] == "256" and f1[2] == "-500"):
            try:
                return int(float(f1[0]))
            except ValueError:
                return 0
    return 0


def load_replay(path: str | Path) -> ReplayFrames:
    """Parse an .osr into absolute-time cursor/key frames."""
    path = Path(path)
    # osrparse needs bytes; some lazer exports store plain lzma stream only?
    try:
        rep: Replay = Replay.from_path(str(path))
    except Exception:
        # lazer sometimes writes replays whose data field must be re-parsed
        rep = _robust_load(path)

    events = rep.replay_data
    if events is None:
        raise ValueError(f"no replay data in {path}")

    times: list[int] = []
    xs: list[float] = []
    ys: list[float] = []
    keys: list[int] = []
    t = _time_base_offset(path)
    for ev in events:
        t += ev.time_delta  # w in the legacy stream is a delta
        # w = -12345 is the seed frame — skip
        if ev.time_delta == -12345 or (hasattr(ev, "w") and getattr(ev, "w", 0) == -12345):
            continue
        x, y = ev.x, ev.y
        # stable sentinel frames: (256, -500)
        if x == 256 and y == -500:
            continue
        times.append(t)
        xs.append(x)
        ys.append(y)
        keys.append(int(ev.keys or 0))

    return ReplayFrames(
        times=np.asarray(times, dtype=np.int64),
        x=np.asarray(xs, dtype=np.float32),
        y=np.asarray(ys, dtype=np.float32),
        keys=np.asarray(keys, dtype=np.uint8),
        mods=getattr(rep, "mods", ()),
        count300=getattr(rep, "count_300", 0) or 0,
        count100=getattr(rep, "count_100", 0) or 0,
        count50=getattr(rep, "count_50", 0) or 0,
        count_miss=getattr(rep, "count_miss", 0) or 0,
        max_combo=getattr(rep, "max_combo", 0) or 0,
        beatmap_hash=getattr(rep, "beatmap_hash", "") or "",
        username=getattr(rep, "username", "") or "",
        game_version=_as_int(getattr(rep, "game_version", 0)),
    )


def _as_int(v) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _robust_load(path: Path) -> Replay:
    raw = path.read_bytes()
    # last-resort: some tools write the lzma frame stream directly
    import lzma

    text = lzma.decompress(raw).decode("ascii", errors="replace")
    events = parse_replay_data(text)
    class _R:  # minimal shim
        pass

    r = _R()
    r.replay_data = events
    r.mods = ()
    r.count_300 = r.count_100 = r.count_50 = r.count_miss = r.max_combo = 0
    r.beatmap_hash = ""
    r.username = path.stem
    return r


BUTTON_BITS = (1, 2, 4, 8)  # M1, M2, K1, K2


def key_edges(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(press_indices, release_indices) of per-button transitions, merged + sorted.

    osu! counts a press for *any* button's 0->1 edge, so an alternating
    K1/K2 stream (one releases exactly as the other presses, no '0' frame
    in between) still yields two presses. Collapsing to "any key held"
    loses those, hence per-bit edges here.

    Raw .osr button bitfield: 1=M1, 2=M2, 4=K1, 8=K2, 16=smoke.
    """
    press_sets: list[np.ndarray] = []
    release_sets: list[np.ndarray] = []
    for bit in BUTTON_BITS:
        down = ((keys & bit) > 0).astype(np.int8)
        press_sets.append(np.flatnonzero(np.diff(np.concatenate(([0], down))) == 1))
        release_sets.append(np.flatnonzero(np.diff(np.concatenate((down, [0]))) == -1))
    press = np.sort(np.concatenate(press_sets)) if press_sets else np.empty(0, np.int64)
    release = np.sort(np.concatenate(release_sets)) if release_sets else np.empty(0, np.int64)
    return press, release


def is_held(keys: np.ndarray) -> np.ndarray:
    """Boolean per frame: any tapping button held."""
    return (keys & 0b1111) > 0


def held_through(frames: ReplayFrames, t0: int, t1: int) -> bool:
    """True if some button stays held for the whole [t0, t1] interval.

    Frames are event-driven (state constant between them), so it's enough
    to check that no frame-time inside the interval shows all buttons up.
    A release inside the window means the slider was dropped.
    """
    if t1 <= t0:
        return True
    lo = int(np.searchsorted(frames.times, t0, side="right"))
    hi = int(np.searchsorted(frames.times, t1, side="left"))
    if hi <= lo:
        return True  # no state change recorded inside — treat as held
    return bool(np.all(is_held(frames.keys[lo:hi])))


def held_at(frames: ReplayFrames, t: int) -> bool:
    """Button state (any tapping button down) at the last frame <= t."""
    idx = int(np.searchsorted(frames.times, t, side="right")) - 1
    if idx < 0:
        return False
    return bool(is_held(frames.keys[idx]))
