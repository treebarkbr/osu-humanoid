"""Faithful Python port of osu!framework geometry used for slider paths.

Sources:
  osu.Framework/Utils/PathApproximator.cs
  osu.Framework/Utils/CircularArcProperties.cs

Vector2 is float in C#; we use Python floats (double) — the resulting
difference in path length is far below one osu!pixel and does not affect
tick/judgement counts.
"""
from __future__ import annotations

import math

BEZIER_TOLERANCE = 0.25
CATMULL_DETAIL = 50
CIRCULAR_ARC_TOLERANCE = 0.1
ALMOST_EQUALS_EPS = 1e-3


class Vec:
    __slots__ = ("x", "y")

    def __init__(self, x: float = 0.0, y: float = 0.0):
        self.x = float(x)
        self.y = float(y)

    def __add__(self, o: "Vec") -> "Vec":
        return Vec(self.x + o.x, self.y + o.y)

    def __sub__(self, o: "Vec") -> "Vec":
        return Vec(self.x - o.x, self.y - o.y)

    def __mul__(self, s: float) -> "Vec":
        return Vec(self.x * s, self.y * s)

    __rmul__ = __mul__

    def __truediv__(self, s: float) -> "Vec":
        return Vec(self.x / s, self.y / s)

    def __neg__(self) -> "Vec":
        return Vec(-self.x, -self.y)

    def __eq__(self, o) -> bool:
        return isinstance(o, Vec) and self.x == o.x and self.y == o.y

    def __repr__(self) -> str:
        return f"({self.x:.4f},{self.y:.4f})"

    def dot(self, o: "Vec") -> float:
        return self.x * o.x + self.y * o.y

    def length(self) -> float:
        return math.hypot(self.x, self.y)

    def length_squared(self) -> float:
        return self.x * self.x + self.y * self.y

    def normalized(self) -> "Vec":
        l = self.length()
        if l == 0:
            return Vec(0.0, 0.0)
        return Vec(self.x / l, self.y / l)


def distance(a: Vec, b: Vec) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def almost_equals(a: float, b: float, eps: float = ALMOST_EQUALS_EPS) -> bool:
    return abs(a - b) < eps


class CircularArcProperties:
    __slots__ = ("is_valid", "theta_start", "theta_range", "direction", "radius", "centre")

    def __init__(self, control_points: list[Vec]):
        a, b, c = control_points[0], control_points[1], control_points[2]

        if almost_equals(0.0, (b.y - a.y) * (c.x - a.x) - (b.x - a.x) * (c.y - a.y)):
            self.is_valid = False
            self.theta_start = 0.0
            self.theta_range = 0.0
            self.direction = 0.0
            self.radius = 0.0
            self.centre = Vec()
            return

        d = 2 * (a.x * (b - c).y + b.x * (c - a).y + c.x * (a - b).y)
        a_sq = a.length_squared()
        b_sq = b.length_squared()
        c_sq = c.length_squared()

        self.centre = Vec(
            (a_sq * (b - c).y + b_sq * (c - a).y + c_sq * (a - b).y) / d,
            (a_sq * (c - b).x + b_sq * (a - c).x + c_sq * (b - a).x) / d,
        )

        d_a = a - self.centre
        d_c = c - self.centre
        self.radius = d_a.length()

        theta_start = math.atan2(d_a.y, d_a.x)
        theta_end = math.atan2(d_c.y, d_c.x)

        while theta_end < theta_start:
            theta_end += 2 * math.pi

        direction = 1.0
        theta_range = theta_end - theta_start

        ortho_a_to_c = c - a
        ortho_a_to_c = Vec(ortho_a_to_c.y, -ortho_a_to_c.x)

        if ortho_a_to_c.dot(b - a) < 0:
            direction = -direction
            theta_range = 2 * math.pi - theta_range

        self.theta_start = theta_start
        self.theta_range = theta_range
        self.direction = direction
        self.is_valid = True


def _clone(cp: list[Vec]) -> list[Vec]:
    return [Vec(v.x, v.y) for v in cp]


def linear_to_piecewise_linear(control_points: list[Vec]) -> list[Vec]:
    return _clone(control_points)


def catmull_find_point(v1: Vec, v2: Vec, v3: Vec, v4: Vec, t: float) -> Vec:
    t2 = t * t
    t3 = t * t2
    return Vec(
        0.5 * (2 * v2.x + (-v1.x + v3.x) * t + (2 * v1.x - 5 * v2.x + 4 * v3.x - v4.x) * t2 + (-v1.x + 3 * v2.x - 3 * v3.x + v4.x) * t3),
        0.5 * (2 * v2.y + (-v1.y + v3.y) * t + (2 * v1.y - 5 * v2.y + 4 * v3.y - v4.y) * t2 + (-v1.y + 3 * v2.y - 3 * v3.y + v4.y) * t3),
    )


def catmull_to_piecewise_linear(control_points: list[Vec]) -> list[Vec]:
    result: list[Vec] = []
    n = len(control_points)
    for i in range(n - 1):
        v1 = control_points[i - 1] if i > 0 else control_points[i]
        v2 = control_points[i]
        v3 = control_points[i + 1] if i < n - 1 else v2 + v2 - v1
        v4 = control_points[i + 2] if i < n - 2 else v3 + v3 - v2
        for c in range(CATMULL_DETAIL):
            result.append(catmull_find_point(v1, v2, v3, v4, c / CATMULL_DETAIL))
            result.append(catmull_find_point(v1, v2, v3, v4, (c + 1) / CATMULL_DETAIL))
    return result


def circular_arc_to_piecewise_linear(control_points: list[Vec]) -> list[Vec]:
    pr = CircularArcProperties(control_points)
    if not pr.is_valid:
        return bezier_to_piecewise_linear(control_points)

    if 2 * pr.radius <= CIRCULAR_ARC_TOLERANCE:
        amount_points = 2
    else:
        amount_points = max(2, math.ceil(pr.theta_range / (2 * math.acos(1 - CIRCULAR_ARC_TOLERANCE / pr.radius))))

    output: list[Vec] = []
    for i in range(amount_points):
        fract = i / (amount_points - 1)
        theta = pr.theta_start + pr.direction * fract * pr.theta_range
        o = Vec(math.cos(theta), math.sin(theta)) * pr.radius
        output.append(pr.centre + o)
    return output


def bezier_to_piecewise_linear(control_points: list[Vec]) -> list[Vec]:
    return bspline_to_piecewise_linear(control_points, max(1, len(control_points) - 1))


def _bspline_to_bezier_internal(control_points: list[Vec], degree: int):
    degree = min(degree, len(control_points) - 1)
    point_count = len(control_points) - 1
    points = _clone(control_points)

    segments: list[list[Vec]] = []

    if degree == point_count:
        segments.append(points)
    else:
        for i in range(point_count - degree):
            sub_bezier = [None] * (degree + 1)
            sub_bezier[0] = points[i]
            for j in range(degree - 1):
                sub_bezier[j + 1] = points[i + 1]
                for k in range(1, degree - j):
                    l = min(k, point_count - degree - i)
                    points[i + k] = (points[i + k] * l + points[i + k + 1]) / (l + 1)
            sub_bezier[degree] = points[i + 1]
            segments.append(sub_bezier)
        segments.append(points[point_count - degree:])

    # returned so that pop() yields segments[0] first (mirrors Stack reversal)
    return list(reversed(segments))


def _bezier_is_flat_enough(control_points: list[Vec]) -> bool:
    for i in range(1, len(control_points) - 1):
        d = control_points[i - 1] - control_points[i] * 2 + control_points[i + 1]
        if d.length_squared() > BEZIER_TOLERANCE * BEZIER_TOLERANCE * 4:
            return False
    return True


def _bezier_subdivide(control_points: list[Vec], l: list, r: list, subdivision_buffer: list, count: int) -> None:
    midpoints = subdivision_buffer
    for i in range(count):
        midpoints[i] = control_points[i]
    for i in range(count):
        l[i] = midpoints[0]
        r[count - i - 1] = midpoints[count - i - 1]
        for j in range(count - i - 1):
            midpoints[j] = (midpoints[j] + midpoints[j + 1]) / 2


def _bezier_approximate(control_points: list[Vec], output: list[Vec], buf1: list, buf2: list, count: int) -> None:
    l = buf2
    r = buf1
    _bezier_subdivide(control_points, l, r, buf1, count)
    for i in range(count - 1):
        l[count + i] = r[i + 1]
    output.append(control_points[0])
    for i in range(1, count - 1):
        index = 2 * i
        p = (l[index - 1] + l[index] * 2 + l[index + 1]) * 0.25
        output.append(p)


def bspline_to_piecewise_linear(control_points: list[Vec], degree: int) -> list[Vec]:
    if len(control_points) < 2:
        return _clone(control_points) if control_points else []

    degree = min(degree, len(control_points) - 1)

    output: list[Vec] = []
    point_count = len(control_points) - 1

    to_flatten = _bspline_to_bezier_internal(control_points, degree)
    free_buffers: list[list[Vec]] = []

    subdivision_buffer1 = [Vec() for _ in range(degree + 1)]
    subdivision_buffer2 = [Vec() for _ in range(degree * 2 + 1)]

    left_child = subdivision_buffer2

    while to_flatten:
        parent = to_flatten.pop()

        if _bezier_is_flat_enough(parent):
            _bezier_approximate(parent, output, subdivision_buffer1, subdivision_buffer2, degree + 1)
            free_buffers.append(parent)
            continue

        right_child = free_buffers.pop() if free_buffers else [Vec() for _ in range(degree + 1)]
        _bezier_subdivide(parent, left_child, right_child, subdivision_buffer1, degree + 1)

        for i in range(degree + 1):
            parent[i] = left_child[i]

        to_flatten.append(right_child)
        to_flatten.append(parent)

    output.append(control_points[point_count])
    return output
