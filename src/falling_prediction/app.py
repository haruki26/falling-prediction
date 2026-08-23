"""Small, injectable webcam application loop."""

from __future__ import annotations

import time

import cv2
import numpy as np

from .calibration import DEFAULT_DESTINATION_RECT, DEFAULT_OUTPUT_SIZE, PerspectiveCalibration, load_calibration, save_calibration
from .config import AppConfig, ConfigurationError
from .openvino_pose import PoseEstimator
from .pose_decoder import decode_poses
from .risk import BedRegion, RiskEvaluator
from .ui import (
    BedBoundary,
    Joint,
    OverlayRenderer,
    PersonSkeleton,
    RiskStatus,
    Telemetry,
)


def run(config: AppConfig, *, capture=None, estimator=None, renderer=None) -> None:
    cap = (
        capture
        if capture is not None
        else cv2.VideoCapture(config.camera_index, cv2.CAP_DSHOW)
    )
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera {config.camera_index}")
    renderer = renderer or OverlayRenderer()
    try:
        explicit = (config.bed_left, config.bed_top, config.bed_right, config.bed_bottom)
        if any(value is not None for value in explicit):
            raise ConfigurationError("explicit rectangular bed overrides are deprecated; use v2 perspective calibration")
        # A forced calibration must work even when the existing file is a
        # legacy v1 rectangle that cannot be rectified.
        calibration = None if config.calibrate else load_calibration(config.calibration_file)
        pending_frame = None
        if calibration is None or config.calibrate:
            observed_shape = None

            def calibration_read():
                nonlocal observed_shape
                result = cap.read()
                if result[0] and result[1] is not None:
                    observed_shape = result[1].shape[:2][::-1]
                return result

            selected = renderer.calibrate_bed_live(calibration_read, initial_region=None)
            if selected is None:
                print("Bed calibration cancelled; exiting.")
                return
            if observed_shape is None:
                # Test doubles and alternate UIs may not read during calibration.
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if hasattr(cap, "get") else 0
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if hasattr(cap, "get") else 0
                if width <= 0 or height <= 0:
                    ok, pending_frame = cap.read()
                    if not ok:
                        raise ConfigurationError("calibration did not provide a source frame")
                    observed_shape = pending_frame.shape[:2][::-1]
                else:
                    observed_shape = (width, height)
            calibration = PerspectiveCalibration(tuple(selected.points), observed_shape[0], observed_shape[1], *DEFAULT_OUTPUT_SIZE, DEFAULT_DESTINATION_RECT)
            save_calibration(config.calibration_file, calibration)
        evaluator = RiskEvaluator(calibration.bed_region)
        if estimator is None:
            if config.model_path is None:
                raise ValueError("model path is required")
            estimator = PoseEstimator(config.model_path, config.device)
        previous = time.perf_counter()
        while True:
            if pending_frame is not None:
                ok, frame = True, pending_frame
                pending_frame = None
            else:
                ok, frame = cap.read()
            if not ok:
                break
            started = time.perf_counter()
            corrected = calibration.rectify(frame)
            pafs, heatmaps = estimator.infer(corrected)
            poses, scores = decode_poses(pafs, heatmaps)
            people = [
                PersonSkeleton(
                    [
                        Joint(i, float(x), float(y), float(c))
                        for i, (x, y, c) in enumerate(p)
                        if np.isfinite(x) and np.isfinite(y) and np.isfinite(c)
                    ]
                )
                for p in poses
            ]
            def in_bed(p: np.ndarray) -> bool:
                torso = p[[5, 6, 11, 12], :2]
                torso_center = (float(np.mean(torso[:, 0])), float(np.mean(torso[:, 1])))
                return bool(
                    np.isfinite(torso_center[0])
                    and np.isfinite(torso_center[1])
                    and evaluator.bed.left <= torso_center[0] <= evaluator.bed.right
                    and evaluator.bed.top <= torso_center[1] <= evaluator.bed.bottom
                )
            selected = (
                next((i for i in np.argsort(scores)[::-1] if in_bed(poses[i])), None)
                if len(poses)
                else None
            )
            if selected is None and len(poses):
                selected = int(np.argmax(scores))
            result = (
                evaluator.evaluate(poses[selected])
                if selected is not None
                else evaluator.evaluate(np.full((17, 3), np.nan))
            )
            elapsed = time.perf_counter() - previous
            previous = time.perf_counter()
            risk = (
                renderer.from_risk_result(result)
                if selected is not None
                else RiskStatus("waiting")
            )
            output = renderer.render(
                corrected,
                Telemetry(
                    fps=1 / max(elapsed, 1e-6),
                    device=config.device,
                    person_count=len(people),
                    inference_ms=(time.perf_counter() - started) * 1000,
                ),
                risk=risk,
                persons=people,
                bed_boundary=renderer.from_bed_region(evaluator.bed),
            )
            renderer.show(output)
            if renderer.should_quit():
                break
    finally:
        cap.release()
        renderer.close()
