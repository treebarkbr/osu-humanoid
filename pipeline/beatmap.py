"""osu!standard (.osu) beatmap parser with full slider geometry.

Ports the relevant parts of:
  osu.Game/Beatmaps/Formats/LegacyBeatmapDecoder.cs
  osu.Game/Rulesets/Objects/Legacy/ConvertHitObjectParser.cs
and attaches a real ``SliderPath`` to every slider so that head/tick/repeat/
tail positions and timings match lazer exactly.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .geometry import Vec
from .hitwindows import (
    object_radius,
    osu_hit_windows,
    precision_adjusted_beat_length,
    approach_time as _approach_time,
)
from . import slider_events as SE
from .slider_path import (
    PATH_BEZIER,
    PATH_CATMULL,
    PATH_LINEAR,
    PATH_PERFECT_CURVE,
    PathControlPoint,
    PathType,
    SliderPath,
)

CIRCLE, SLIDER, SPINNER = 1, 2, 8
FIRST_LAZER_VERSION = 128
MAX_COORDINATE_VALUE = 512.0


@dataclass
class HitObject:
    time: int
    x: float
    y: float
    type_bits: int
    end_time: int = 0
    is_circle: bool = False
    is_slider: bool = False
    is_spinner: bool = False
    # slider-only
    path: "SliderPath | None" = None
    span_count: int = 1
    span_duration: float = 0.0
    velocity: float = 0.0
    tick_distance: float = 0.0
    generate_ticks: bool = True

    @property
    def is_clickable(self) -> bool:
        return self.is_circle or self.is_slider

    @property
    def position(self) -> Vec:
        return Vec(self.x, self.y)


@dataclass
class TimingPoint:
    time: float
    beat_length: float
    uninherited: bool


@dataclass
class Beatmap:
    path: str = ""
    format_version: int = 14
    version: str = ""
    artist: str = ""
    title: str = ""
    creator: str = ""
    beatmap_id: int = 0
    beatmap_set_id: int = 0
    mode: int = 0
    cs: float = 4.0
    od: float = 5.0
    ar: float = 5.0
    hp: float = 5.0
    slider_multiplier: float = 1.4
    slider_tick_rate: float = 1.0
    stack_leniency: float = 0.7
    hit_objects: list[HitObject] = field(default_factory=list)
    timing_points: list[TimingPoint] = field(default_factory=list)

    @property
    def circles(self) -> list[HitObject]:
        return [o for o in self.hit_objects if o.is_circle]

    @property
    def clickables(self) -> list[HitObject]:
        return [o for o in self.hit_objects if o.is_clickable]

    @property
    def radius(self) -> float:
        return object_radius(self.cs)


def parse_beatmap(path: str | Path) -> Beatmap:
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    bm = Beatmap(path=str(path))

    m = re.search(r"osu file format v(\d+)", lines[0] if lines else "")
    bm.format_version = int(m.group(1)) if m else 14

    section = ""
    general: dict[str, str] = {}
    metadata: dict[str, str] = {}
    difficulty: dict[str, str] = {}
    raw_objects: list[str] = []

    for raw in lines:
        line = raw.strip().lstrip("\ufeff")
        if not line or line.startswith("//"):
            continue
        sm = re.match(r"\[(\w+)\]", line)
        if sm:
            section = sm.group(1)
            continue
        if section == "HitObjects":
            raw_objects.append(line)
            continue
        if section == "TimingPoints":
            parts = line.split(",")
            try:
                t = float(parts[0].strip())
                bl = float(parts[1].strip())
                uninh = len(parts) < 7 or parts[6].strip()[0] == "1"
                bm.timing_points.append(TimingPoint(t, bl, uninh))
            except (ValueError, IndexError):
                pass
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if section == "General":
            general[key] = value
        elif section == "Metadata":
            metadata[key] = value
        elif section == "Difficulty":
            difficulty[key] = value

    bm.mode = int(general.get("Mode", "0"))
    try:
        bm.stack_leniency = float(general.get("StackLeniency", "0.7"))
    except (TypeError, ValueError):
        bm.stack_leniency = 0.7
    bm.version = metadata.get("Version", "")
    bm.artist = metadata.get("Artist", "")
    bm.title = metadata.get("Title", "")
    bm.creator = metadata.get("Creator", "")
    bm.beatmap_id = int(metadata.get("BeatmapID", "0") or 0)
    bm.beatmap_set_id = int(metadata.get("BeatmapSetID", "0") or 0)

    def _f(key: str, default: float) -> float:
        try:
            return float(difficulty.get(key, default))
        except (TypeError, ValueError):
            return default

    bm.cs = min(max(_f("CircleSize", 4.0), 0.0), 10.0)
    bm.od = min(max(_f("OverallDifficulty", 5.0), 0.0), 10.0)
    bm.ar = min(max(_f("ApproachRate", bm.od), 0.0), 10.0)
    bm.hp = min(max(_f("HPDrainRate", 5.0), 0.0), 10.0)
    bm.slider_multiplier = min(max(_f("SliderMultiplier", 1.4), 0.4), 3.6)
    bm.slider_tick_rate = min(max(_f("SliderTickRate", 1.0), 0.5), 8.0)

    bm.timing_points.sort(key=lambda p: p.time)

    for line in raw_objects:
        _parse_hit_object(bm, line)
    bm.hit_objects.sort(key=lambda o: o.time)
    return bm


def _parse_hit_object(bm: Beatmap, line: str) -> None:
    parts = line.split(",")
    if len(parts) < 4:
        return
    try:
        x = float(parts[0])
        y = float(parts[1])
        t = int(float(parts[2]))
        type_bits = int(parts[3])
    except ValueError:
        return

    x = min(max(x, 0.0), 512.0)
    y = min(max(y, 0.0), 512.0)
    if bm.format_version < FIRST_LAZER_VERSION:
        x, y = float(int(x)), float(int(y))

    obj = HitObject(time=t, x=x, y=y, type_bits=type_bits)

    if type_bits & CIRCLE:
        obj.is_circle = True
        obj.end_time = t
    elif type_bits & SLIDER:
        obj.is_slider = True
        _build_slider(bm, obj, parts)
    elif type_bits & SPINNER:
        obj.is_spinner = True
        try:
            obj.end_time = int(max(0.0, float(parts[5])))
        except (IndexError, ValueError):
            obj.end_time = t
    else:
        return
    bm.hit_objects.append(obj)


# --- slider construction ---------------------------------------------
def _timing_context(bm: Beatmap, t: float) -> tuple[float, float, bool]:
    """(beat_length, slider_velocity_multiplier, generate_ticks) at time t."""
    beat_length = 1000.0
    sv = 1.0
    generate_ticks = True
    for tp in bm.timing_points:
        if tp.time <= t + 1e-3:
            if tp.uninherited:
                beat_length = tp.beat_length
            sv = 100.0 / -tp.beat_length if tp.beat_length < 0 else 1.0
            generate_ticks = tp.beat_length == tp.beat_length  # not NaN
        else:
            break
    return beat_length, sv, generate_ticks


def _build_slider(bm: Beatmap, obj: HitObject, parts: list[str]) -> None:
    obj.is_slider = True
    start = Vec(obj.x, obj.y)

    try:
        repeats_field = int(parts[6])
    except (IndexError, ValueError):
        repeats_field = 1
    span_count = max(1, repeats_field)

    pixel_length = None
    if len(parts) > 7:
        try:
            pixel_length = max(0.0, float(parts[7]))
        except ValueError:
            pixel_length = None
        if pixel_length == 0:
            pixel_length = None

    point_string = parts[5] if len(parts) > 5 else ""
    control_points = _convert_path_string(point_string, start, bm.format_version)

    path = SliderPath(control_points, pixel_length, optimise_catmull=True)

    # zero-length slider: reset repeats (matches ConvertHitObjectParser.createSlider)
    if abs(path.distance) < 1e-3:
        span_count = 1

    beat_length, sv, generate_ticks = _timing_context(bm, obj.time)
    adjusted = precision_adjusted_beat_length(beat_length, sv)
    velocity = 100.0 * bm.slider_multiplier / adjusted if adjusted > 0 else 0.0

    obj.path = path
    obj.span_count = span_count
    obj.velocity = velocity
    obj.span_duration = path.distance / velocity if velocity > 0 else 0.0

    if generate_ticks and obj.span_duration > 0 and not (isinstance(sv, float) and sv != sv):
        scoring_distance = velocity * beat_length
        obj.tick_distance = scoring_distance / bm.slider_tick_rate
    else:
        obj.tick_distance = float("inf")
    obj.generate_ticks = generate_ticks

    obj.end_time = int(round(obj.time + span_count * obj.span_duration))


def _convert_path_type(token: str) -> PathType:
    c = token[0]
    if c == "L":
        return PATH_LINEAR
    if c == "P":
        return PATH_PERFECT_CURVE
    if c == "B":
        if len(token) > 1:
            try:
                degree = int(token[1:])
                if degree > 0:
                    return PathType("BSpline", degree)
            except ValueError:
                pass
        return PATH_BEZIER
    return PATH_CATMULL


def _read_point(value: str, start: Vec, fmt: int) -> Vec:
    a = value.split(":")
    if fmt >= FIRST_LAZER_VERSION:
        px, py = float(a[0]), float(a[1])
    else:
        px, py = float(int(float(a[0]))), float(int(float(a[1])))
    return Vec(px - start.x, py - start.y)


def _convert_path_string(point_string: str, start: Vec, fmt: int) -> list[PathControlPoint]:
    split = point_string.split("|")
    points: list[Vec] = []
    segments: list[tuple[PathType, int]] = []

    for s in split:
        if not s:
            continue
        if s[0].isalpha():
            segments.append((_convert_path_type(s), len(points)))
            if len(points) == 0:
                points.append(Vec(0.0, 0.0))
        else:
            points.append(_read_point(s, start, fmt))

    control: list[PathControlPoint] = []
    for i, (ptype, start_index) in enumerate(segments):
        if i < len(segments) - 1:
            end_index = segments[i + 1][1]
            seg_points = points[start_index:end_index]
            control += _convert_points(ptype, seg_points, points[end_index], fmt)
        else:
            control += _convert_points(ptype, points[start_index:], None, fmt)
    return control


def _convert_points(ptype: PathType, pts: list[Vec], end_point: Vec | None, fmt: int) -> list[PathControlPoint]:
    vertices = [PathControlPoint(Vec(p.x, p.y)) for p in pts]

    if ptype.type == "PerfectCurve":
        end_point_length = 0 if end_point is None else 1
        if fmt < FIRST_LAZER_VERSION:
            if len(vertices) + end_point_length != 3:
                ptype = PATH_BEZIER
            elif _is_linear(pts[0], pts[1], end_point if end_point is not None else pts[2]):
                ptype = PATH_LINEAR
        elif len(vertices) + end_point_length > 3:
            ptype = PATH_BEZIER

    vertices[0].type = ptype

    out: list[PathControlPoint] = []
    start_index = 0
    end_index = 0
    n = len(vertices)

    while True:
        end_index += 1
        if end_index >= n:
            break
        if vertices[end_index].position != vertices[end_index - 1].position:
            continue
        if ptype.type == "Catmull" and end_index > 1 and fmt < FIRST_LAZER_VERSION:
            continue
        if end_index == n - 1:
            continue
        vertices[end_index - 1].type = ptype
        out += vertices[start_index:end_index]
        start_index = end_index + 1

    if start_index < end_index:
        out += vertices[start_index:end_index]
    return out


def _is_linear(p0: Vec, p1: Vec, p2: Vec) -> bool:
    from .geometry import almost_equals
    return almost_equals(0.0, (p1.y - p0.y) * (p2.x - p0.x) - (p1.x - p0.x) * (p2.y - p0.y))


# --- public helpers ---------------------------------------------------
def slider_events(bm: Beatmap, obj: HitObject) -> list[SE.SliderEventDescriptor]:
    if not obj.is_slider or obj.path is None:
        return []
    return SE.generate(
        float(obj.time),
        obj.span_duration,
        obj.velocity,
        obj.tick_distance,
        obj.path.distance,
        obj.span_count,
    )


def curve_position(obj: HitObject, progress: float) -> Vec:
    """Absolute playfield position on the slider path at `progress`."""
    if obj.path is None:
        return Vec(obj.x, obj.y)
    p = obj.path.position_at(progress)
    return Vec(obj.x + p.x, obj.y + p.y)


def apply_mod_difficulty(bm: Beatmap, *, hr: bool = False, ez: bool = False) -> tuple[float, float, float]:
    cs, ar, od = bm.cs, bm.ar, bm.od
    if hr:
        cs = min(10.0, cs * 1.3)
        ar = min(10.0, ar * 1.4)
        od = min(10.0, od * 1.4)
    if ez:
        cs *= 0.5
        ar *= 0.5
        od *= 0.5
    return cs, ar, od


def hit_windows(od: float) -> tuple[float, float, float]:
    """Legacy helper: (w300, w100, w50) half-widths."""
    g, o, m, _ = osu_hit_windows(od)
    return g, o, m


def circle_radius(cs: float) -> float:
    return object_radius(cs)


def approach_time(ar: float) -> float:
    return _approach_time(ar)
