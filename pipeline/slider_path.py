"""Faithful Python port of osu.Game SliderPath + legacy path control points.

Sources:
  osu.Game/Rulesets/Objects/SliderPath.cs
  osu.Game/Rulesets/Objects/PathControlPoint.cs
  osu.Game/Rulesets/Objects/Types/PathType.cs
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

from . import geometry as G
from .geometry import Vec

CATMULL, BSPLINE, LINEAR, PERFECT_CURVE = "Catmull", "BSpline", "Linear", "PerfectCurve"

CATMULL_DETAIL = 50


@dataclass
class PathType:
    type: str
    degree: int | None = None


PATH_CATMULL = PathType(CATMULL)
PATH_BEZIER = PathType(BSPLINE, None)
PATH_LINEAR = PathType(LINEAR)
PATH_PERFECT_CURVE = PathType(PERFECT_CURVE)


@dataclass
class PathControlPoint:
    position: Vec
    type: PathType | None = None


class SliderPath:
    """Mirrors SliderPath. ExpectedDistance is used to shorten/lengthen."""

    def __init__(self, control_points: list[PathControlPoint], expected_distance: float | None = None,
                 optimise_catmull: bool = True):
        self.control_points = control_points
        self.expected_distance = expected_distance
        self.optimise_catmull = optimise_catmull

        self._calculated_path: list[Vec] = []
        self._cumulative_length: list[float] = []
        self._calculated_length = 0.0
        self._optimised_length = 0.0
        self._segment_ends: list[int] = []
        self._segment_end_distances: list[float] = []
        self._valid = False

    # -- public ---------------------------------------------------------
    @property
    def distance(self) -> float:
        self._ensure_valid()
        return self._cumulative_length[-1] if self._cumulative_length else 0.0

    @property
    def calculated_distance(self) -> float:
        self._ensure_valid()
        return self._calculated_length

    @property
    def calculated_path(self) -> list[Vec]:
        self._ensure_valid()
        return self._calculated_path

    def position_at(self, progress: float) -> Vec:
        self._ensure_valid()
        d = self._progress_to_distance(progress)
        return self._interpolate_vertices(self._index_of_distance(d), d)

    def get_segment_ends(self) -> list[float]:
        self._ensure_valid()
        dist = self.distance
        return [x / dist for x in self._segment_end_distances] if dist else []

    # -- internals ------------------------------------------------------
    def _ensure_valid(self) -> None:
        if self._valid:
            return
        self._calculate_path()
        self._calculate_length()
        self._valid = True

    def _calculate_path(self) -> None:
        self._calculated_path = []
        self._segment_ends = []
        self._optimised_length = 0.0

        if not self.control_points:
            return

        vertices = [cp.position for cp in self.control_points]
        start = 0

        for i in range(len(self.control_points)):
            if self.control_points[i].type is None and i < len(self.control_points) - 1:
                continue

            segment_vertices = vertices[start:i + 1]
            segment_type = self.control_points[start].type or PATH_LINEAR

            if len(segment_vertices) == 1:
                self._calculated_path.append(segment_vertices[0])
            elif len(segment_vertices) > 1:
                sub_path = self._calculate_sub_path(segment_vertices, segment_type)
                skip_first = (self._calculated_path and sub_path and self._calculated_path[-1] == sub_path[0])
                for j in range(1 if skip_first else 0, len(sub_path)):
                    self._calculated_path.append(sub_path[j])

            if i > 0:
                self._segment_ends.append(len(self._calculated_path) - 1)

            start = i

    def _calculate_sub_path(self, sub_control_points: list[Vec], type_: PathType) -> list[Vec]:
        if type_.type == LINEAR:
            return G.linear_to_piecewise_linear(sub_control_points)

        if type_.type == PERFECT_CURVE:
            if len(sub_control_points) == 3:
                pr = G.CircularArcProperties(sub_control_points)
                if pr.is_valid:
                    if 2 * pr.radius <= 0.1:
                        sub_points = 2
                    else:
                        sub_points = max(2, math.ceil(pr.theta_range / (2.0 * math.acos(1 - (0.1 / pr.radius)))))
                    if sub_points < 1000:
                        sub_path = G.circular_arc_to_piecewise_linear(sub_control_points)
                        if sub_path:
                            return sub_path

        elif type_.type == CATMULL:
            sub_path = G.catmull_to_piecewise_linear(sub_control_points)
            if not self.optimise_catmull:
                return sub_path

            optimised_path: list[Vec] = []
            last_start: Vec | None = None
            length_removed_since_start = 0.0

            for i in range(len(sub_path)):
                if last_start is None:
                    optimised_path.append(sub_path[i])
                    last_start = sub_path[i]
                    continue
                dist_from_start = G.distance(last_start, sub_path[i])
                length_removed_since_start += G.distance(sub_path[i - 1], sub_path[i])

                catmull_segment_length = CATMULL_DETAIL * 2
                if dist_from_start > 6 or (i + 1) % catmull_segment_length == 0 or i == len(sub_path) - 1:
                    optimised_path.append(sub_path[i])
                    self._optimised_length += length_removed_since_start - dist_from_start
                    last_start = None
                    length_removed_since_start = 0.0

            return optimised_path

        degree = type_.degree if type_.degree is not None else len(sub_control_points)
        return G.bspline_to_piecewise_linear(sub_control_points, degree)

    def _calculate_length(self) -> None:
        self._calculated_length = self._optimised_length
        self._cumulative_length = [0.0]

        path = self._calculated_path
        for i in range(len(path) - 1):
            self._calculated_length += (path[i + 1] - path[i]).length()
            self._cumulative_length.append(self._calculated_length)

        self._segment_end_distances = [self._cumulative_length[e] for e in self._segment_ends]

        exp = self.expected_distance
        if exp is not None and self._calculated_length != exp:
            if len(path) >= 2 and path[-1] == path[-2] and exp > self._calculated_length:
                self._cumulative_length.append(self._calculated_length)
                return

            self._cumulative_length.pop()
            path_end_index = len(path) - 1

            if self._calculated_length > exp:
                while self._cumulative_length and self._cumulative_length[-1] >= exp:
                    self._cumulative_length.pop()
                    path.pop()
                    path_end_index -= 1

            if path_end_index <= 0:
                self._cumulative_length.append(0.0)
                return

            direction = (path[path_end_index] - path[path_end_index - 1]).normalized()
            path[path_end_index] = path[path_end_index - 1] + direction * (exp - self._cumulative_length[-1])
            self._cumulative_length.append(exp)

    def _index_of_distance(self, d: float) -> int:
        return bisect.bisect_left(self._cumulative_length, d)

    def _progress_to_distance(self, progress: float) -> float:
        return min(1.0, max(0.0, progress)) * self.distance

    def _interpolate_vertices(self, i: int, d: float) -> Vec:
        path = self._calculated_path
        if not path:
            return Vec()
        if i <= 0:
            return path[0]
        if i >= len(path):
            return path[-1]

        p0 = path[i - 1]
        p1 = path[i]
        d0 = self._cumulative_length[i - 1]
        d1 = self._cumulative_length[i]

        if G.almost_equals(d0, d1):
            return p0

        w = (d - d0) / (d1 - d0)
        return p0 + (p1 - p0) * w
