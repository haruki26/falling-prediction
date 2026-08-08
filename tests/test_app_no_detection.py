import numpy as np

from falling_prediction.app import run
from falling_prediction.config import AppConfig


class Capture:
    def __init__(self): self.frames = [np.zeros((16, 16, 3), np.uint8)]; self.released = False
    def isOpened(self): return True
    def read(self): return (True, self.frames.pop()) if self.frames else (False, None)
    def release(self): self.released = True


class Estimator:
    def infer(self, frame):
        return np.zeros((1, 38, 32, 57)), np.zeros((1, 19, 32, 57))


class Renderer:
    def __init__(self, calibration=None): self.statuses = []; self.calibration = calibration; self.calibration_calls = 0; self.closed = False
    def render(self, frame, telemetry, *, risk, persons, bed_boundary):
        self.statuses.append(risk.level); return frame
    def show(self, frame): pass
    def should_quit(self): return True
    def close(self): self.closed = True
    def calibrate_bed_live(self, read_frame, *, initial_region=None):
        self.calibration_calls += 1
        if self.calibration is None: return None
        from falling_prediction.ui import BedBoundary
        return self.calibration
    from_risk_result = staticmethod(lambda result: None)
    from_bed_region = staticmethod(lambda region: None)


def test_no_detection_waits_instead_of_reporting_safe():
    renderer = Renderer()
    run(
        AppConfig(
            model_path=None,
            bed_left=0.1,
            bed_top=0.1,
            bed_right=0.9,
            bed_bottom=0.9,
        ),
        capture=Capture(),
        estimator=Estimator(),
        renderer=renderer,
    )
    assert renderer.statuses == ["waiting"]


def test_saved_roi_skips_live_calibration(tmp_path):
    from falling_prediction.config import save_bed_region
    save_bed_region(tmp_path / "roi.json", (0.1, 0.2, 0.8, 0.9))
    renderer = Renderer()
    run(AppConfig(model_path=None, calibration_file=tmp_path / "roi.json"),
        capture=Capture(), estimator=Estimator(), renderer=renderer)
    assert renderer.calibration_calls == 0


def test_live_calibration_saves_roi(tmp_path):
    from falling_prediction.ui import BedBoundary
    path = tmp_path / "roi.json"
    renderer = Renderer(BedBoundary([(0.2, 0.3), (0.7, 0.3), (0.7, 0.8), (0.2, 0.8)]))
    run(AppConfig(model_path=None, calibration_file=path),
        capture=Capture(), estimator=Estimator(), renderer=renderer)
    assert renderer.calibration_calls == 1
    assert path.exists()


def test_live_calibration_cancel_releases_capture(tmp_path):
    capture = Capture()
    renderer = Renderer(None)
    run(AppConfig(model_path=None, calibration_file=tmp_path / "roi.json"),
        capture=capture, estimator=Estimator(), renderer=renderer)
    assert capture.released and renderer.closed
