"""osu! mod parsing + their effect on osu!standard geometry/timing.

Only mods that change the judged playfield matter:
  HR  -> vertical flip, CS/AR/OD/HP scaled (CS*1.3, AR/OD/HP*1.4, capped at 10)
  EZ  -> CS/AR/OD/HP halved
  DT / NC -> clock rate 1.5x
  HT  -> clock rate 0.75x
  HD / FL / SD / PF / NF / TD / RX / AP / SO -> no effect on hit geometry/timing

Clock rate scales *times* (object times and approach duration), not the real-time
hit windows (OD is unchanged by DT/HT), matching osu! behaviour.
"""
from __future__ import annotations

# stable mod bitflags
NF, EZ, TD, HD, HR, SD, DT, RX = 1, 2, 4, 8, 16, 32, 64, 128
HT, NC, FL, AT, SO, AP, PF = 256, 512, 1024, 2048, 4096, 8192, 16384

_BIT_ACRONYM = {
    "NF": NF, "EZ": EZ, "TD": TD, "HD": HD, "HR": HR, "SD": SD, "DT": DT,
    "RX": RX, "HT": HT, "NC": NC, "FL": FL, "AT": AT, "SO": SO, "AP": AP, "PF": PF,
}

ACRONYMS = ["NF", "EZ", "TD", "HD", "HR", "SD", "DT", "RX", "HT", "NC", "FL",
            "AT", "SO", "AP", "PF", "4K", "5K", "6K", "7K", "8K", "9K", "CO", "V2"]


def _bits_int(mods) -> int:
    if mods is None:
        return 0
    if isinstance(mods, int):
        return mods
    val = getattr(mods, "value", None)
    if isinstance(val, int):
        return val
    if isinstance(mods, str):
        return acronyms_to_bits(mods)
    # iterable of acronyms / Mod
    total = 0
    for m in mods:
        if isinstance(m, int):
            total |= m
        else:
            total |= _bit_of(getattr(m, "name", str(m)))
    return total


def _bit_of(acr: str) -> int:
    return _BIT_ACRONYM.get(acr.upper(), 0)


def acronyms_to_bits(s: str) -> int:
    s = s.strip()
    if not s:
        return 0
    total = 0
    if " " in s or len(s) % 2 == 0:
        i = 0
        while i < len(s):
            acr = s[i:i + 2].upper()
            if acr in _BIT_ACRONYM:
                total |= _BIT_ACRONYM[acr]
                i += 2
            else:
                i += 1
    else:
        total |= _bit_of(s)
    return total


def bits_to_acronyms(bits: int) -> list[str]:
    return [a for a in ACRONYMS if bits & _BIT_ACRONYM.get(a, 0)]


def parse_mods(mods) -> int:
    """Normalise anything (str / int / osrparse Mod / iterable) to a bitmask."""
    return _bits_int(mods)


def is_hr(mods) -> bool:
    return bool(parse_mods(mods) & HR)


def is_ez(mods) -> bool:
    return bool(parse_mods(mods) & EZ)


def clock_rate(mods) -> float:
    b = parse_mods(mods)
    if b & DT or b & NC:
        return 1.5
    if b & HT:
        return 0.75
    return 1.0


def effective_difficulty(cs: float, ar: float, od: float, hp: float, mods) -> tuple[float, float, float, float]:
    b = parse_mods(mods)
    if b & HR:
        cs, ar, od, hp = min(10.0, cs * 1.3), min(10.0, ar * 1.4), min(10.0, od * 1.4), min(10.0, hp * 1.4)
    if b & EZ:
        cs, ar, od, hp = cs * 0.5, ar * 0.5, od * 0.5, hp * 0.5
    return cs, ar, od, hp


def mods_str(mods) -> str:
    return "".join(bits_to_acronyms(parse_mods(mods)))
