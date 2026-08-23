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


@dataclass(frozen=True)
class BedRegion:
    left: float = 0.10
    top: float = 0.10
    right: float = 0.90
    bottom: float = 0.90

    def __post_init__(self) -> None:
        if not all(
            np.isfinite(v) for v in (self.left, self.top, self.right, self.bottom)
        ) or not (
            0 <= self.left < self.right <= 1 and 0 <= self.top < self.bottom <= 1
        ):
            raise ValueError("invalid normalized bed rectangle")


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
        edge = (
            min(self.bed.right - self.bed.left, self.bed.bottom - self.bed.top) * 0.12
        )
        key_body = xy[[0, 11, 12]]
        near_edge = any(
            np.isfinite(p).all()
            and (
                p[0] <= self.bed.left + edge
                or p[0] >= self.bed.right - edge
                or p[1] <= self.bed.top + edge
                or p[1] >= self.bed.bottom - edge
            )
            for p in key_body
        )
        outside = any(
            np.isfinite(p).all()
            and (
                p[0] < self.bed.left
                or p[0] > self.bed.right
                or p[1] < self.bed.top
                or p[1] > self.bed.bottom
            )
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

                def edge_distance(p: Iterable[float]) -> float:
                    x, y = p
                    return min(
                        x - self.bed.left,
                        self.bed.right - x,
                        y - self.bed.top,
                        self.bed.bottom - y,
                    )

                toward_edge = edge_distance(center) < edge_distance(previous) - 1e-6
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
