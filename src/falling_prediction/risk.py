"""Deterministic, explainable fall-risk rules over pose and recent movement."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from enum import IntEnum
from math import atan2, degrees, hypot

import numpy as np


# This is an image-plane orientation heuristic, not a measurement of physical
# elevation; its meaning depends on the camera's orientation.  In normalized
# image coordinates, a torso within 35 degrees of the image vertical is raised.
UPPER_BODY_RAISED_MAX_ANGLE_DEGREES = 35.0
MIN_TORSO_LENGTH = 0.08


class RiskLevel(IntEnum):
    SAFE = 0
    CAUTION = 1
    DANGER = 2


@dataclass(frozen=True)
class RiskResult:
    level: RiskLevel
    score: int
    reasons: tuple[str, ...]


@dataclass(frozen=True, init=False)
class BedRegion:
    """A validated, normalized, simple four-vertex bed polygon.

    The positional four-number form and ``left=...`` keyword form are retained
    as a small compatibility shim for version-1 rectangle configurations.
    """

    points: tuple[tuple[float, float], ...]

    def __init__(self, *args, points=None, left=None, top=None, right=None, bottom=None):
        if points is not None:
            if args or any(v is not None for v in (left, top, right, bottom)):
                raise TypeError("specify either points or rectangle coordinates")
            raw = points
        elif args:
            if len(args) != 4:
                raise TypeError("rectangle form requires four coordinates")
            left, top, right, bottom = args
            raw = ((left, top), (right, top), (right, bottom), (left, bottom))
        elif any(v is not None for v in (left, top, right, bottom)):
            if not all(v is not None for v in (left, top, right, bottom)):
                raise ValueError("all rectangle coordinates are required")
            raw = ((left, top), (right, top), (right, bottom), (left, bottom))
        else:
            raw = ((0.10, 0.10), (0.90, 0.10), (0.90, 0.90), (0.10, 0.90))
        try:
            normalized = tuple((float(p[0]), float(p[1])) for p in raw)
        except (TypeError, IndexError, ValueError) as exc:
            raise ValueError("bed polygon must contain four coordinate pairs") from exc
        if len(normalized) != 4 or not all(np.isfinite(p).all() for p in normalized):
            raise ValueError("bed polygon must contain four finite points")
        if not all(0 <= x <= 1 and 0 <= y <= 1 for x, y in normalized):
            raise ValueError("bed polygon coordinates must be normalized to 0..1")
        if len(set(normalized)) != 4 or abs(_polygon_area(normalized)) <= 1e-10:
            raise ValueError("bed polygon must be non-degenerate")
        if _segments_cross(normalized[0], normalized[1], normalized[2], normalized[3]) or _segments_cross(
            normalized[1], normalized[2], normalized[3], normalized[0]
        ):
            raise ValueError("bed polygon must not self-intersect")
        object.__setattr__(self, "points", normalized)

    @property
    def left(self) -> float: return min(p[0] for p in self.points)
    @property
    def top(self) -> float: return min(p[1] for p in self.points)
    @property
    def right(self) -> float: return max(p[0] for p in self.points)
    @property
    def bottom(self) -> float: return max(p[1] for p in self.points)

    def contains(self, point: Iterable[float]) -> bool:
        x, y = point
        inside = False
        for a, b in zip(self.points, self.points[1:] + self.points[:1]):
            if _point_segment_distance((x, y), a, b) <= 1e-10:
                return True
            if (a[1] > y) != (b[1] > y) and x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0]:
                inside = not inside
        return inside

    def distance_to_edges(self, point: Iterable[float]) -> float:
        return min(_point_segment_distance(point, a, b) for a, b in zip(self.points, self.points[1:] + self.points[:1]))

    @property
    def edge_threshold(self) -> float:
        return min(hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(self.points, self.points[1:] + self.points[:1])) * 0.12


def _polygon_area(points) -> float:
    return 0.5 * sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(points, points[1:] + points[:1]))


def _segments_cross(a, b, c, d) -> bool:
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    def on_segment(p, q, r):
        return (min(p[0], r[0]) - 1e-12 <= q[0] <= max(p[0], r[0]) + 1e-12
                and min(p[1], r[1]) - 1e-12 <= q[1] <= max(p[1], r[1]) + 1e-12)
    ab_c, ab_d = orient(a, b, c), orient(a, b, d)
    cd_a, cd_b = orient(c, d, a), orient(c, d, b)
    if ab_c * ab_d < -1e-24 and cd_a * cd_b < -1e-24:
        return True
    return ((abs(ab_c) <= 1e-12 and on_segment(a, c, b))
            or (abs(ab_d) <= 1e-12 and on_segment(a, d, b))
            or (abs(cd_a) <= 1e-12 and on_segment(c, a, d))
            or (abs(cd_b) <= 1e-12 and on_segment(c, b, d)))


def _point_segment_distance(p, a, b) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    length2 = dx * dx + dy * dy
    t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length2))
    return hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy))


class RiskEvaluator:
    """Evaluate one person using normalized keypoints and a bounded history.

    Keypoint indices follow COCO: nose=0, shoulders=5/6, hips=11/12,
    knees=13/14, ankles=15/16. Inputs are ``(17, 3)`` arrays.
    """

    def __init__(self, bed: BedRegion | None = None, history_size: int = 8) -> None:
        self.bed = bed if bed is not None else BedRegion()
        self.history: deque[tuple[float, float]] = deque(maxlen=history_size)

    def evaluate(self, keypoints: Iterable[Iterable[float]]) -> RiskResult:
        points = np.asarray(keypoints, dtype=float)
        if points.shape != (17, 3):
            raise ValueError("keypoints must have shape (17, 3): x, y, confidence")
        valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0.0)
        xy = np.where(valid[:, None], points[:, :2], np.nan)
        center_points = xy[[5, 6, 11, 12]]
        center = (
            np.nanmean(center_points, axis=0)
            if np.isfinite(center_points).all(axis=1).any()
            else np.array([np.nan, np.nan])
        )
        reasons: list[str] = []
        score = 0
        key_body = xy[[0, 11, 12]]
        near_edge = any(
            np.isfinite(p).all()
            and self.bed.distance_to_edges(p) <= self.bed.edge_threshold
            for p in key_body
        )
        outside = any(
            np.isfinite(p).all() and not self.bed.contains(p)
            for p in xy
        )
        if outside:
            score += 3
            reasons.append("body part outside bed")
        if near_edge:
            score += 2
            reasons.append("head or hip near bed edge")
        shoulders = xy[[5, 6]]
        hips = xy[[11, 12]]
        upright = False
        if np.isfinite(shoulders).all() and np.isfinite(hips).all():
            shoulder_midpoint = np.mean(shoulders, axis=0)
            hip_midpoint = np.mean(hips, axis=0)
            torso_vector = hip_midpoint - shoulder_midpoint
            torso_length = float(np.linalg.norm(torso_vector))
            if torso_length >= MIN_TORSO_LENGTH:
                # atan2(dx, dy) measures signed deviation from image vertical;
                # use its absolute value because lean direction is irrelevant.
                torso_angle_degrees = abs(
                    degrees(atan2(float(torso_vector[0]), float(torso_vector[1])))
                )
                upright = torso_angle_degrees <= UPPER_BODY_RAISED_MAX_ANGLE_DEGREES
        if upright:
            score += 1
            reasons.append("upper body raised")
        if np.isfinite(center).all():
            if self.history:
                previous = self.history[-1]
                movement = hypot(center[0] - previous[0], center[1] - previous[1])

                toward_edge = self.bed.distance_to_edges(center) < self.bed.distance_to_edges(previous) - 1e-6
                if movement >= 0.06 and toward_edge:
                    score += 2
                    reasons.append("rapid movement toward edge")
            self.history.append((float(center[0]), float(center[1])))
        else:
            self.history.clear()
        level = (
            RiskLevel.DANGER
            if score >= 3
            else RiskLevel.CAUTION
            if score >= 1
            else RiskLevel.SAFE
        )
        return RiskResult(level, score, tuple(reasons))
