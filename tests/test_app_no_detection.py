import numpy as np
import pytest

from falling_prediction.app import run
from falling_prediction.calibration import PerspectiveCalibration, load_calibration, save_calibration
from falling_prediction.config import AppConfig, ConfigurationError


class Capture:
    def __init__(self, count=2):
        self.frames = [np.zeros((16, 16, 3), np.uint8) for _ in range(count)]
        self.released = False
    def isOpened(self): return True
    def read(self): return (True, self.frames.pop()) if self.frames else (False, None)
    def release(self): self.released = True


class Estimator:
    def __init__(self): self.infer_shapes = []
    def infer(self, frame):
        self.infer_shapes.append(frame.shape)
        return np.zeros((1, 38, 32, 57)), np.zeros((1, 19, 32, 57))


class Renderer:
    def __init__(self, calibration=None):
        self.statuses = []
        self.calibration = calibration
        self.calibration_calls = 0
        self.closed = False
        self.rendered_shapes = []
        self.boundaries = []
    def render(self, frame, telemetry, *, risk, persons, bed_boundary):
        self.statuses.append(risk.level)
        self.rendered_shapes.append(frame.shape)
        self.boundaries.append(bed_boundary.points)
        return frame
    def show(self, frame): pass
    def should_quit(self): return True
    def close(self): self.closed = True
    def calibrate_bed_live(self, read_frame, *, initial_region=None):
        self.calibration_calls += 1
        return self.calibration
    from_risk_result = staticmethod(lambda result: None)
    @staticmethod
    def from_bed_region(region):
        from falling_prediction.ui import BedBoundary
        return BedBoundary([(region.left, region.top), (region.right, region.top),
                            (region.right, region.bottom), (region.left, region.bottom)])


def test_no_detection_waits_instead_of_reporting_safe(tmp_path):
    path = tmp_path / "calibration.json"
    save_calibration(path, PerspectiveCalibration(
        ((.1, .1), (.9, .1), (.9, .9), (.1, .9)), 16, 16
    ))
    renderer = Renderer()
    run(
        AppConfig(
            model_path=None,
            calibration_file=path,
        ),
        capture=Capture(count=2),
        estimator=Estimator(),
        renderer=renderer,
    )
    assert renderer.statuses == ["waiting"]


def test_saved_roi_skips_live_calibration(tmp_path):
    path = tmp_path / "calibration.json"
    save_calibration(path, PerspectiveCalibration(
        ((.1, .1), (.9, .1), (.9, .9), (.1, .9)), 16, 16
    ))
    renderer = Renderer()
    run(AppConfig(model_path=None, calibration_file=path),
        capture=Capture(), estimator=Estimator(), renderer=renderer)
    assert renderer.calibration_calls == 0
    assert renderer.rendered_shapes == [(256, 456, 3)]


def test_live_calibration_saves_v2_and_rectifies_before_estimator_and_render(tmp_path):
    from falling_prediction.ui import BedBoundary
    path = tmp_path / "calibration.json"
    renderer = Renderer(BedBoundary([(0.2, 0.3), (0.7, 0.3), (0.7, 0.8), (0.2, 0.8)]))
    estimator = Estimator()
    run(AppConfig(model_path=None, calibration_file=path),
        capture=Capture(count=2), estimator=estimator, renderer=renderer)
    assert renderer.calibration_calls == 1
    assert path.exists()
    saved = load_calibration(path)
    assert saved is not None and saved.source_frame_dimensions == (16, 16)
    assert estimator.infer_shapes == [(256, 456, 3)]
    assert renderer.rendered_shapes == [(256, 456, 3)]
    assert renderer.boundaries[0] == [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)]


def test_live_calibration_cancel_releases_capture(tmp_path):
    capture = Capture()
    renderer = Renderer(None)
    run(AppConfig(model_path=None, calibration_file=tmp_path / "roi.json"),
        capture=capture, estimator=Estimator(), renderer=renderer)
    assert capture.released and renderer.closed


def test_deprecated_rectangle_overrides_fail_clearly():
    with pytest.raises(ConfigurationError, match="deprecated"):
        run(AppConfig(model_path=None, bed_left=0.1, bed_top=0.1,
                      bed_right=0.9, bed_bottom=0.9),
            capture=Capture(), estimator=Estimator(), renderer=Renderer())


def test_v1_calibration_requires_recalibration(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text('{"version": 1, "left": 0.1, "top": 0.1, "right": 0.9, "bottom": 0.9}')
    with pytest.raises(ConfigurationError, match="recalibration"):
        run(AppConfig(model_path=None, calibration_file=path),
            capture=Capture(), estimator=Estimator(), renderer=Renderer())
    assert path.exists()
