"""Perspective mapping from the camera view into bed-local coordinates."""

from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np

UNIT_SQUARE = np.array(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), dtype=np.float32)


def compute_homography(points: Iterable[Iterable[float]]) -> np.ndarray:
    """Map ordered normalized camera corners (UL, UR, LR, LL) to a unit square."""
    try:
        source = np.asarray(tuple(points), dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("homography requires four finite coordinate pairs") from exc
    if source.shape != (4, 2):
        raise ValueError("homography requires exactly four coordinate pairs")
    if not np.isfinite(source).all():
        raise ValueError("homography points must be finite")
    if ((source < 0) | (source > 1)).any():
        raise ValueError("homography points must be normalized to 0..1")
    if (
        len(np.unique(source, axis=0)) != 4
        or abs(cv2.contourArea(source)) <= 1e-10
        or not cv2.isContourConvex(source)
    ):
        raise ValueError("homography points must form a non-degenerate quadrilateral")
    try:
        matrix = cv2.getPerspectiveTransform(source, UNIT_SQUARE)
    except cv2.error as exc:
        raise ValueError(f"could not construct homography: {exc}") from exc
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("constructed homography is not finite")
    return matrix


build_homography = compute_homography
create_homography = compute_homography


def transform_keypoints(keypoints: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """Project a ``(17, 3)`` pose, preserving confidence and invalid rows."""
    points = np.asarray(keypoints)
    if points.shape != (17, 3):
        raise ValueError("keypoints must have shape (17, 3): x, y, confidence")
    matrix = np.asarray(homography, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("homography must be a finite 3x3 matrix")
    result = np.array(points, dtype=float, copy=True)
    valid = np.isfinite(points).all(axis=1)
    if valid.any():
        try:
            projected = cv2.perspectiveTransform(
                np.asarray(points[valid, :2], dtype=np.float32).reshape(-1, 1, 2), matrix
            ).reshape(-1, 2)
        except cv2.error as exc:
            raise ValueError(f"could not transform keypoints: {exc}") from exc
        if not np.isfinite(projected).all():
            raise ValueError("homography produced non-finite keypoint coordinates")
        result[valid, :2] = projected
    return result


def transform_poses(poses: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """Project an array of ``(17, 3)`` poses."""
    array = np.asarray(poses)
    # ``decode_poses`` returns ``np.asarray([])`` when no person was detected.
    if array.size == 0:
        return np.empty((0, 17, 3), dtype=float)
    if array.ndim != 3 or array.shape[1:] != (17, 3):
        raise ValueError("poses must have shape (N, 17, 3)")
    return np.stack([transform_keypoints(pose, homography) for pose in array])


transform_pose = transform_keypoints
