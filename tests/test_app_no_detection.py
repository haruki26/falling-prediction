import numpy as np

from falling_prediction.app import run
from falling_prediction.config import AppConfig


class Capture:
    def __init__(self): self.frames = [np.zeros((16, 16, 3), np.uint8)]
    def isOpened(self): return True
    def read(self): return (True, self.frames.pop()) if self.frames else (False, None)
    def release(self): pass


class Estimator:
    def infer(self, frame):
        return np.zeros((1, 38, 32, 57)), np.zeros((1, 19, 32, 57))


class Renderer:
    def __init__(self): self.statuses = []
    def render(self, frame, telemetry, *, risk, persons, bed_boundary):
        self.statuses.append(risk.level); return frame
    def show(self, frame): pass
    def should_quit(self): return True
    def close(self): pass
    from_risk_result = staticmethod(lambda result: None)
    from_bed_region = staticmethod(lambda region: None)


def test_no_detection_waits_instead_of_reporting_safe():
    renderer = Renderer()
    run(AppConfig(model_path=None), capture=Capture(), estimator=Estimator(), renderer=renderer)
    assert renderer.statuses == ["waiting"]
