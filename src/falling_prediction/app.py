"""Small, injectable webcam application loop."""

from __future__ import annotations

import time

import cv2
import numpy as np

from .config import AppConfig, load_bed_region, save_bed_region
from .openvino_pose import PoseEstimator
from .pose_decoder import decode_poses
from .risk import BedRegion, RiskEvaluator
from .ui import BedBoundary, Joint, OverlayRenderer, PersonSkeleton, RiskStatus, Telemetry


def run(config: AppConfig, *, capture=None, estimator=None, renderer=None) -> None:
    cap = capture if capture is not None else cv2.VideoCapture(config.camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera {config.camera_index}")
    renderer = renderer or OverlayRenderer()
    try:
        explicit = (config.bed_left, config.bed_top, config.bed_right, config.bed_bottom)
        if all(value is not None for value in explicit):
            assert all(value is not None for value in explicit)
            region_values = tuple(float(value) for value in explicit)  # type: ignore[arg-type]
        else:
            saved = load_bed_region(config.calibration_file)
            if saved is None or config.calibrate:
                initial = (
                    BedBoundary(points=[(saved[0], saved[1]), (saved[2], saved[1]),
                                         (saved[2], saved[3]), (saved[0], saved[3])])
                    if saved is not None else None
                )
                selected = renderer.calibrate_bed_live(cap.read, initial_region=initial)
                if selected is None:
                    print("Bed calibration cancelled; exiting.")
                    return
                xs = [point[0] for point in selected.points]
                ys = [point[1] for point in selected.points]
                region_values = (min(xs), min(ys), max(xs), max(ys))
                save_bed_region(config.calibration_file, region_values)
            else:
                region_values = saved
        evaluator = RiskEvaluator(BedRegion(*region_values))
        if estimator is None:
            if config.model_path is None:
                raise ValueError("model path is required")
            estimator = PoseEstimator(config.model_path, config.device)
        previous = time.perf_counter()
        while True:
            ok, frame = cap.read()
            if not ok: break
            started = time.perf_counter()
            pafs, heatmaps = estimator.infer(frame)
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
            in_bed = lambda p: (
                np.isfinite(p[[5, 6, 11, 12], :2]).all()
                and evaluator.bed.left
                <= np.mean(p[[5, 6, 11, 12], 0])
                <= evaluator.bed.right
                and evaluator.bed.top
                <= np.mean(p[[5, 6, 11, 12], 1])
                <= evaluator.bed.bottom
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
                frame,
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
