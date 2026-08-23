"""Perspective calibration and image rectification.

The persisted format is deliberately independent of the UI.  UI code supplies
four normalized points in clockwise order: LT, RT, RB, LB.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


DEFAULT_OUTPUT_SIZE = (456, 256)
DEFAULT_DESTINATION_RECT = (0.10, 0.10, 0.90, 0.90)


def _points(points) -> tuple[tuple[float, float], ...]:
    try:
        result = tuple((float(x), float(y)) for x, y in points)
    except (TypeError, ValueError) as exc:
        raise ValueError("source_points must contain four (x, y) points") from exc
    if len(result) != 4 or not all(math.isfinite(v) for p in result for v in p):
        raise ValueError("source_points must contain four finite points")
    if not all(0 <= x <= 1 and 0 <= y <= 1 for x, y in result):
        raise ValueError("source_points must be normalized to 0..1")
    area = sum(result[i][0] * result[(i + 1) % 4][1] - result[(i + 1) % 4][0] * result[i][1] for i in range(4)) / 2
    if area <= 1e-8:
        raise ValueError("source_points must form a non-degenerate clockwise polygon")
    crosses = []
    for i in range(4):
        ax, ay = result[i]
        bx, by = result[(i + 1) % 4]
        cx, cy = result[(i + 2) % 4]
        crosses.append((bx - ax) * (cy - by) - (by - ay) * (cx - bx))
    if not all(c > 1e-8 for c in crosses):
        raise ValueError("source_points must be four ordered corners (LT, RT, RB, LB)")
    return result


@dataclass(frozen=True)
class PerspectiveCalibration:
    source_points: tuple[tuple[float, float], ...]
    source_width: int
    source_height: int
    output_width: int = DEFAULT_OUTPUT_SIZE[0]
    output_height: int = DEFAULT_OUTPUT_SIZE[1]
    destination_bed_rect: tuple[float, float, float, float] = DEFAULT_DESTINATION_RECT

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_points", _points(self.source_points))
        if self.source_width <= 0 or self.source_height <= 0:
            raise ValueError("source frame dimensions must be positive")
        if self.output_width <= 0 or self.output_height <= 0:
            raise ValueError("output dimensions must be positive")
        l, t, r, b = (float(v) for v in self.destination_bed_rect)
        if not all(math.isfinite(v) for v in (l, t, r, b)) or not (0 <= l < r <= 1 and 0 <= t < b <= 1):
            raise ValueError("destination_bed_rect must be a normalized rectangle")
        object.__setattr__(self, "destination_bed_rect", (l, t, r, b))

    @property
    def source_frame_dimensions(self) -> tuple[int, int]:
        return (self.source_width, self.source_height)

    @property
    def output_dimensions(self) -> tuple[int, int]:
        return (self.output_width, self.output_height)

    @property
    def destination_rect(self) -> tuple[float, float, float, float]:
        return self.destination_bed_rect

    @property
    def bed_region(self):
        from .risk import BedRegion
        return BedRegion(*self.destination_bed_rect)

    def rectify(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("camera frame must be HWC BGR")
        h, w = frame.shape[:2]
        if (w, h) != (self.source_width, self.source_height):
            raise ValueError(f"camera frame is {w}x{h}; calibration requires {self.source_width}x{self.source_height}")
        src = np.asarray(self.source_points, dtype=np.float32) * np.array([w - 1, h - 1], dtype=np.float32)
        l, t, r, b = self.destination_bed_rect
        dst = np.asarray([(l, t), (r, t), (r, b), (l, b)], dtype=np.float32) * np.array([self.output_width - 1, self.output_height - 1], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(frame, matrix, (self.output_width, self.output_height))


def load_calibration(path: Path) -> PerspectiveCalibration | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") == 1:
            raise ValueError("legacy v1 rectangular calibration; recalibration is required (file retained)")
        if data.get("version") != 2:
            raise ValueError("unsupported calibration version; recalibration is required")
        size = data["source_frame"]
        output = data.get("output_dimensions", {"width": 456, "height": 256})
        rect = data["destination_bed_rect"]
        rect = tuple(rect[k] for k in ("left", "top", "right", "bottom"))
        return PerspectiveCalibration(tuple(map(tuple, data["source_points"])), int(size["width"]), int(size["height"]), int(output["width"]), int(output["height"]), rect)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        from .config import ConfigurationError
        raise ConfigurationError(f"invalid calibration file {path}: {exc}") from exc


def save_calibration(path: Path, calibration: PerspectiveCalibration) -> None:
    payload = {"version": 2, "source_points": [list(p) for p in calibration.source_points],
               "source_frame": {"width": calibration.source_width, "height": calibration.source_height},
               "output_dimensions": {"width": calibration.output_width, "height": calibration.output_height},
               "destination_bed_rect": dict(zip(("left", "top", "right", "bottom"), calibration.destination_bed_rect))}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
