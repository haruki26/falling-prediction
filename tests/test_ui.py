"""Tests for the OpenCV monitoring overlay (ui.py).

These tests avoid touching capture, inference, or risk calculation.  When
OpenCV is not installed, a minimal fake module is injected so rendering logic
can still be exercised.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import numpy as np
import pytest


class _FakeCv2(ModuleType):
    """Minimal cv2 stand-in for tests that exercise drawing logic."""

    FONT_HERSHEY_SIMPLEX = 0
    LINE_AA = 16
    WND_PROP_VISIBLE = 1
    INTER_LINEAR = 1

    EVENT_LBUTTONDOWN = 1
    EVENT_MOUSEMOVE = 0
    EVENT_LBUTTONUP = 4

    WINDOW_NORMAL = 1

    class error(Exception):
        pass

    def __init__(self) -> None:
        super().__init__("cv2")
        self._last_key = -1
        self._window_visible: dict[str, float] = {}
        self._mouse_callbacks: dict[str, tuple[Any, Any]] = {}
        self._mouse_event_queue: list[tuple[str, int, int, int, int]] = []
        self._key_queue: list[int] = []

    def imshow(self, winname: str, mat: Any) -> None:
        self._window_visible[winname] = 1.0

    def namedWindow(self, winname: str, flags: int = 0) -> None:
        self._window_visible[winname] = 1.0

    def setMouseCallback(self, winname: str, onMouse: Any, param: Any = None) -> None:
        self._mouse_callbacks[winname] = (onMouse, param)

    def waitKey(self, delay: int = 0) -> int:
        # Dispatch at most one queued mouse event per frame, then return the
        # next queued key (falling back to the single last-key value).
        if self._mouse_event_queue:
            winname, event, x, y, flags = self._mouse_event_queue.pop(0)
            callback, param = self._mouse_callbacks.get(winname, (None, None))
            if callback is not None:
                callback(event, x, y, flags, param)
        if self._key_queue:
            return self._key_queue.pop(0)
        return self._last_key

    def set_next_key(self, key: int) -> None:
        self._last_key = key

    def queue_key(self, key: int) -> None:
        self._key_queue.append(key)

    def queue_mouse(self, winname: str, event: int, x: int, y: int, flags: int = 0) -> None:
        self._mouse_event_queue.append((winname, event, x, y, flags))

    def getWindowProperty(self, winname: str, prop: int) -> float:
        return self._window_visible.get(winname, 1.0)

    def destroyWindow(self, winname: str) -> None:
        self._window_visible.pop(winname, None)
        self._mouse_callbacks.pop(winname, None)

    def getTextSize(self, text: str, fontFace: int, fontScale: float, thickness: int) -> tuple[tuple[int, int], int]:
        # Rough approximation so panel sizing stays deterministic.
        return ((len(text) * max(6, int(fontScale * 12)), int(fontScale * 24) + 4), 4)

    def rectangle(self, img: Any, pt1: Any, pt2: Any, color: Any, thickness: int = -1, lineType: int | None = None) -> None:
        pass

    def circle(self, img: Any, center: Any, radius: int, color: Any, thickness: int = -1, lineType: int | None = None) -> None:
        pass

    def line(self, img: Any, pt1: Any, pt2: Any, color: Any, thickness: int = 1, lineType: int | None = None) -> None:
        pass

    def putText(self, img: Any, text: str, org: Any, fontFace: int, fontScale: float, color: Any, thickness: int = 1, lineType: int | None = None) -> None:
        pass

    def addWeighted(self, src1: Any, alpha: float, src2: Any, beta: float, gamma: float, dst: Any) -> None:
        pass

    def fillPoly(self, img: Any, pts: Any, color: Any) -> None:
        pass

    def resize(
        self,
        src: Any,
        dsize: tuple[int, int],
        dst: Any = None,
        fx: float | None = None,
        fy: float | None = None,
        interpolation: int | None = None,
    ) -> Any:
        height, width = dsize[1], dsize[0]
        if src.ndim == 3:
            return np.zeros((height, width, src.shape[2]), dtype=src.dtype)
        return np.zeros((height, width), dtype=src.dtype)


# Install the fake before importing the module under test.  This keeps the
# normal ``import cv2`` path deterministic without coupling tests to the
# implementation's module alias.
sys.modules["cv2"] = _FakeCv2()

from falling_prediction import ui
from falling_prediction.risk import BedRegion, RiskEvaluator, RiskLevel, RiskResult


@pytest.fixture
def fake_opencv(monkeypatch) -> _FakeCv2:
    fake = _FakeCv2()
    monkeypatch.setitem(sys.modules, "cv2", fake)
    monkeypatch.setattr(ui, "cv2", fake)
    return fake


@pytest.fixture
def blank_frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Data contracts and conversion helpers
# ---------------------------------------------------------------------------


def test_dataclasses_are_usable():
    joint = ui.Joint(index=0, x=0.5, y=0.5, confidence=0.9)
    skeleton = ui.PersonSkeleton(joints=[joint])
    boundary = ui.BedBoundary(points=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    risk = ui.RiskStatus(level="safe", reasons=("stable",), score=0)
    telemetry = ui.Telemetry(fps=30.0, device="CPU", person_count=1, inference_ms=16.0)

    assert joint.confidence == pytest.approx(0.9)
    assert skeleton.track_id is None
    assert boundary.label is None
    assert risk.level == "safe"
    assert telemetry.person_count == 1


def test_from_risk_result_maps_levels_and_reasons():
    result = RiskResult(level=RiskLevel.DANGER, score=4, reasons=("body part outside bed", "upper body raised"))
    status = ui.OverlayRenderer.from_risk_result(result)

    assert status.level == "danger"
    assert status.score == 4
    assert "Body part outside bed" in status.reasons
    assert "Upper body raised" in status.reasons


def test_from_risk_result_preserves_unknown_reasons():
    result = RiskResult(level=RiskLevel.CAUTION, score=1, reasons=("custom reason",))
    status = ui.OverlayRenderer.from_risk_result(result)
    assert "custom reason" in status.reasons


def test_from_bed_region_builds_polygon():
    region = BedRegion(left=0.1, top=0.2, right=0.8, bottom=0.9)
    boundary = ui.OverlayRenderer.from_bed_region(region)

    assert boundary.points == [(0.1, 0.2), (0.8, 0.2), (0.8, 0.9), (0.1, 0.9)]
    assert boundary.label == "BED"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_render_preserves_frame_shape(blank_frame):
    renderer = ui.OverlayRenderer()
    telemetry = ui.Telemetry(fps=29.7, device="CPU", person_count=1)
    out = renderer.render(blank_frame, telemetry)

    assert out.shape == blank_frame.shape
    assert out.dtype == blank_frame.dtype
    # Input frame must not be modified.
    assert np.all(blank_frame == 0)


def test_render_rejects_invalid_frame():
    renderer = ui.OverlayRenderer()
    telemetry = ui.Telemetry()

    with pytest.raises(ValueError, match="3-channel BGR image"):
        renderer.render(np.zeros((480, 640), dtype=np.uint8), telemetry)


def test_render_with_risk_levels(blank_frame):
    renderer = ui.OverlayRenderer()
    telemetry = ui.Telemetry(person_count=1)

    for level in ("safe", "caution", "danger", "waiting"):
        out = renderer.render(
            blank_frame,
            telemetry,
            risk=ui.RiskStatus(level=level, reasons=("test reason",), score=1),
        )
        assert out.shape == blank_frame.shape


def test_render_draws_bed_boundary_and_skeleton(blank_frame):
    renderer = ui.OverlayRenderer()
    telemetry = ui.Telemetry(person_count=1)
    bed = ui.BedBoundary(points=[(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)], label="bed")
    joints = [
        ui.Joint(i, x=0.5 + i * 0.01, y=0.5 + i * 0.005, confidence=0.9)
        for i in range(17)
    ]
    person = ui.PersonSkeleton(joints=joints)

    out = renderer.render(blank_frame, telemetry, persons=[person], bed_boundary=bed)
    assert out.shape == blank_frame.shape


def test_render_with_no_persons(blank_frame):
    renderer = ui.OverlayRenderer()
    telemetry = ui.Telemetry(fps=15.0, device="GPU", person_count=0, inference_ms=33.3)
    out = renderer.render(blank_frame, telemetry)
    assert out.shape == blank_frame.shape


def test_custom_labels_override_defaults(blank_frame):
    renderer = ui.OverlayRenderer(labels={"safe": "OK", "fps": "FPS"})
    telemetry = ui.Telemetry(fps=30.0, person_count=0)
    out = renderer.render(blank_frame, telemetry, risk=ui.RiskStatus(level="safe"))
    assert out.shape == blank_frame.shape


def test_default_labels_are_ascii_english():
    assert ui.DEFAULT_LABELS["safe"] == "SAFE"
    assert ui.DEFAULT_LABELS["caution"] == "CAUTION"
    assert ui.DEFAULT_LABELS["danger"] == "DANGER"
    assert ui.DEFAULT_LABELS["waiting"] == "WAITING"
    assert ui.DEFAULT_LABELS["device"] == "DEVICE"
    assert ui.DEFAULT_LABELS["person_count"] == "PEOPLE"
    assert ui.DEFAULT_LABELS["inference"] == "INFERENCE"
    assert ui.DEFAULT_LABELS["bed"] == "BED"
    assert ui.DEFAULT_LABELS["reason"] == "REASONS"
    assert ui.DEFAULT_LABELS["window_title"] == "Fall Risk Monitor"


def test_render_skips_nonfinite_and_low_confidence_joints(blank_frame):
    renderer = ui.OverlayRenderer()
    joints = [
        ui.Joint(0, x=float("nan"), y=0.5, confidence=0.9),
        ui.Joint(1, x=float("inf"), y=0.5, confidence=0.9),
        ui.Joint(2, x=0.5, y=float("-inf"), confidence=0.9),
        ui.Joint(3, x=0.5, y=0.5, confidence=0.1),
        ui.Joint(4, x=0.6, y=0.6, confidence=0.5),
    ]
    person = ui.PersonSkeleton(joints=joints)
    out = renderer.render(
        blank_frame, ui.Telemetry(person_count=1), persons=[person]
    )
    assert out.shape == blank_frame.shape


def test_render_skips_nonfinite_bed_points(blank_frame):
    renderer = ui.OverlayRenderer()
    bed = ui.BedBoundary(
        points=[(0.1, 0.1), (float("nan"), 0.1), (0.9, 0.9), (0.1, 0.9)],
        label="bed",
    )
    out = renderer.render(
        blank_frame, ui.Telemetry(person_count=0), bed_boundary=bed
    )
    assert out.shape == blank_frame.shape


# ---------------------------------------------------------------------------
# Window helpers
# ---------------------------------------------------------------------------


def test_should_quit_on_esc(fake_opencv):
    renderer = ui.OverlayRenderer()
    fake_opencv.set_next_key(27)
    assert renderer.should_quit() is True


def test_should_quit_continues_by_default(fake_opencv):
    renderer = ui.OverlayRenderer()
    fake_opencv.set_next_key(-1)
    assert renderer.should_quit() is False


def test_close_destroys_window(fake_opencv):
    renderer = ui.OverlayRenderer()
    renderer._window_created = True
    fake_opencv.imshow(renderer.window_name, np.zeros((10, 10, 3), dtype=np.uint8))
    renderer.close()
    assert renderer._window_created is False


# ---------------------------------------------------------------------------
# Integration with the real risk evaluator
# ---------------------------------------------------------------------------


def test_end_to_end_with_risk_evaluator(blank_frame):
    evaluator = RiskEvaluator()
    points = np.zeros((17, 3), dtype=float)
    points[:, :2] = (0.5, 0.5)
    points[:, 2] = 1.0

    result = evaluator.evaluate(points)
    status = ui.OverlayRenderer.from_risk_result(result)
    bed = ui.OverlayRenderer.from_bed_region(evaluator.bed)

    renderer = ui.OverlayRenderer()
    telemetry = ui.Telemetry(fps=30.0, device="CPU", person_count=1, inference_ms=20.0)
    out = renderer.render(blank_frame, telemetry, risk=status, bed_boundary=bed)
    assert out.shape == blank_frame.shape
    assert status.level == "safe"


# ---------------------------------------------------------------------------
# Bed corner calibration
# ---------------------------------------------------------------------------


CALIB_WIN = "Calibration Test"
LIVE_CALIB_WIN = "Live Calibration Test"


def _queue_four_corners(fake, win, corners):
    for x, y in corners:
        fake.queue_mouse(win, fake.EVENT_LBUTTONDOWN, x, y, 0)


def test_calibrate_bed_rejects_invalid_frame():
    renderer = ui.OverlayRenderer()
    with pytest.raises(ValueError, match="3-channel BGR image"):
        renderer.calibrate_bed(np.zeros((480, 640), dtype=np.uint8))


def test_calibrate_bed_returns_selection_on_confirm(blank_frame, fake_opencv):
    renderer = ui.OverlayRenderer()
    corners = [(100, 100), (540, 100), (540, 380), (100, 380)]
    _queue_four_corners(fake_opencv, CALIB_WIN, corners)
    for _ in range(3):
        fake_opencv.queue_key(0)
    fake_opencv.queue_key(13)  # Enter

    result = renderer.calibrate_bed(blank_frame, window_name=CALIB_WIN)

    assert result is not None
    assert len(result.points) == 4
    expected = [(x / 640, y / 480) for x, y in corners]
    for actual, exp in zip(result.points, expected):
        assert actual[0] == pytest.approx(exp[0], abs=1e-6)
        assert actual[1] == pytest.approx(exp[1], abs=1e-6)
    assert result.label == "BED"


def test_calibrate_bed_returns_none_on_cancel(blank_frame, fake_opencv):
    renderer = ui.OverlayRenderer()
    fake_opencv.queue_key(27)  # Esc

    result = renderer.calibrate_bed(blank_frame, window_name=CALIB_WIN)

    assert result is None


def test_calibrate_bed_cleans_up_window(blank_frame, fake_opencv):
    renderer = ui.OverlayRenderer()
    fake_opencv.queue_key(27)

    renderer.calibrate_bed(blank_frame, window_name=CALIB_WIN)

    assert CALIB_WIN not in fake_opencv._window_visible
    assert CALIB_WIN not in fake_opencv._mouse_callbacks


def test_calibrate_bed_uses_initial_region(blank_frame, fake_opencv):
    renderer = ui.OverlayRenderer()
    initial = ui.BedBoundary(
        points=[(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]
    )
    fake_opencv.queue_key(13)  # Enter

    result = renderer.calibrate_bed(
        blank_frame, initial_region=initial, window_name=CALIB_WIN
    )

    assert result is not None
    assert len(result.points) == 4
    for actual, exp in zip(result.points, initial.points):
        assert actual[0] == pytest.approx(exp[0], abs=1e-6)
        assert actual[1] == pytest.approx(exp[1], abs=1e-6)


def test_calibrate_bed_reset_restores_initial_region(blank_frame, fake_opencv):
    renderer = ui.OverlayRenderer()
    initial = ui.BedBoundary(
        points=[(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]
    )
    # Click a stray corner, reset, then confirm.
    fake_opencv.queue_mouse(CALIB_WIN, fake_opencv.EVENT_LBUTTONDOWN, 10, 10, 0)
    fake_opencv.queue_key(0)
    fake_opencv.queue_key(ord("r"))
    fake_opencv.queue_key(13)

    result = renderer.calibrate_bed(
        blank_frame, initial_region=initial, window_name=CALIB_WIN
    )

    assert result is not None
    for actual, exp in zip(result.points, initial.points):
        assert actual[0] == pytest.approx(exp[0], abs=1e-6)
        assert actual[1] == pytest.approx(exp[1], abs=1e-6)


def test_calibrate_bed_uses_custom_label(blank_frame, fake_opencv):
    renderer = ui.OverlayRenderer(labels={"bed": "MAT"})
    initial = ui.BedBoundary(
        points=[(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]
    )
    fake_opencv.queue_key(13)

    result = renderer.calibrate_bed(
        blank_frame, initial_region=initial, window_name=CALIB_WIN
    )

    assert result is not None
    assert result.label == "MAT"


def test_calibrate_bed_backspace_undo(blank_frame, fake_opencv):
    renderer = ui.OverlayRenderer()
    corners = [(100, 100), (540, 100), (540, 380), (100, 380)]
    _queue_four_corners(fake_opencv, CALIB_WIN, corners)
    # Add the replacement corner that will be clicked after the undo.
    fake_opencv.queue_mouse(CALIB_WIN, fake_opencv.EVENT_LBUTTONDOWN, 100, 380, 0)
    # Pair one key with each mouse event; backspace lands on the 4th click.
    for _ in range(3):
        fake_opencv.queue_key(0)
    fake_opencv.queue_key(8)  # Backspace
    fake_opencv.queue_key(13)  # Enter

    result = renderer.calibrate_bed(blank_frame, window_name=CALIB_WIN)

    assert result is not None
    expected = [(x / 640, y / 480) for x, y in corners]
    for actual, exp in zip(result.points, expected):
        assert actual[0] == pytest.approx(exp[0], abs=1e-6)
        assert actual[1] == pytest.approx(exp[1], abs=1e-6)


def test_calibrate_bed_ignores_confirm_until_four_valid_corners(blank_frame, fake_opencv):
    renderer = ui.OverlayRenderer()
    # Only three corners, then Enter, then Esc.
    for x, y in [(100, 100), (540, 100), (540, 380)]:
        fake_opencv.queue_mouse(CALIB_WIN, fake_opencv.EVENT_LBUTTONDOWN, x, y, 0)
    for _ in range(2):
        fake_opencv.queue_key(0)
    fake_opencv.queue_key(13)
    fake_opencv.queue_key(27)

    result = renderer.calibrate_bed(blank_frame, window_name=CALIB_WIN)

    assert result is None


def test_calibrate_bed_rejects_invalid_geometry(blank_frame, fake_opencv):
    renderer = ui.OverlayRenderer()
    # Bow-tie / crossing order: LT, RT, LB, RB.
    invalid_corners = [(100, 100), (540, 100), (100, 380), (540, 380)]
    _queue_four_corners(fake_opencv, CALIB_WIN, invalid_corners)
    for _ in range(3):
        fake_opencv.queue_key(0)
    fake_opencv.queue_key(13)  # Enter should be ignored
    fake_opencv.queue_key(27)  # Esc cancels

    result = renderer.calibrate_bed(blank_frame, window_name=CALIB_WIN)

    assert result is None


# ---------------------------------------------------------------------------
# Live bed corner calibration
# ---------------------------------------------------------------------------


def test_calibrate_bed_live_rejects_non_callable():
    renderer = ui.OverlayRenderer()
    with pytest.raises(TypeError, match="read_frame must be callable"):
        renderer.calibrate_bed_live("not callable")  # type: ignore[arg-type]


def test_calibrate_bed_live_renders_multiple_frames(blank_frame, fake_opencv):
    renderer = ui.OverlayRenderer()
    frames = [blank_frame.copy() for _ in range(5)]
    call_count = [0]

    def read_frame() -> tuple[bool, np.ndarray]:
        idx = call_count[0]
        call_count[0] += 1
        if idx < len(frames):
            return True, frames[idx]
        return False, np.zeros((480, 640, 3), dtype=np.uint8)

    # Queue "no key" for a few iterations so multiple frames are rendered,
    # then confirm the initial region.
    for _ in range(3):
        fake_opencv.queue_key(0)
    fake_opencv.queue_key(13)

    initial = ui.BedBoundary(
        points=[(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]
    )
    result = renderer.calibrate_bed_live(
        read_frame, initial_region=initial, window_name=LIVE_CALIB_WIN
    )

    assert result is not None
    assert call_count[0] >= 3
    assert LIVE_CALIB_WIN not in fake_opencv._window_visible
    assert LIVE_CALIB_WIN not in fake_opencv._mouse_callbacks


def test_calibrate_bed_live_confirms_four_clicks(blank_frame, fake_opencv):
    renderer = ui.OverlayRenderer()

    def read_frame() -> tuple[bool, np.ndarray]:
        return True, blank_frame

    # Let a couple of live frames render, then click four corners and confirm.
    for _ in range(2):
        fake_opencv.queue_key(0)
    corners = [(100, 100), (540, 100), (540, 380), (100, 380)]
    _queue_four_corners(fake_opencv, LIVE_CALIB_WIN, corners)
    for _ in range(3):
        fake_opencv.queue_key(0)
    fake_opencv.queue_key(13)

    result = renderer.calibrate_bed_live(read_frame, window_name=LIVE_CALIB_WIN)

    assert result is not None
    expected = [(x / 640, y / 480) for x, y in corners]
    for actual, exp in zip(result.points, expected):
        assert actual[0] == pytest.approx(exp[0], abs=1e-6)
        assert actual[1] == pytest.approx(exp[1], abs=1e-6)


def test_calibrate_bed_live_handles_temporary_read_failure(blank_frame, fake_opencv):
    renderer = ui.OverlayRenderer()
    results = [False, True]
    idx = [0]

    def read_frame() -> tuple[bool, np.ndarray]:
        i = idx[0]
        idx[0] += 1
        if i < len(results):
            return (results[i], blank_frame) if results[i] else (False, blank_frame)
        return True, blank_frame

    initial = ui.BedBoundary(
        points=[(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]
    )
    fake_opencv.queue_key(0)
    fake_opencv.queue_key(13)

    result = renderer.calibrate_bed_live(
        read_frame, initial_region=initial, window_name=LIVE_CALIB_WIN
    )

    assert result is not None
    assert idx[0] >= 2
    for actual, exp in zip(result.points, initial.points):
        assert actual[0] == pytest.approx(exp[0], abs=1e-6)
        assert actual[1] == pytest.approx(exp[1], abs=1e-6)


def test_calibrate_bed_live_returns_none_on_cancel(blank_frame, fake_opencv):
    renderer = ui.OverlayRenderer()

    def read_frame() -> tuple[bool, np.ndarray]:
        return True, blank_frame

    fake_opencv.queue_key(27)

    result = renderer.calibrate_bed_live(read_frame, window_name=LIVE_CALIB_WIN)

    assert result is None
    assert LIVE_CALIB_WIN not in fake_opencv._window_visible
    assert LIVE_CALIB_WIN not in fake_opencv._mouse_callbacks
