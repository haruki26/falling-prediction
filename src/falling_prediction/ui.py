"""OpenCV monitoring overlay for the pose-detection MVP.

This module is intentionally separate from capture, inference, and risk logic.
It accepts plain data objects (telemetry, skeletons, bed boundary, risk status)
and draws a legible overlay on a frame.  The annotated frame can then be shown
with ``cv2.imshow``.

The default labels are ASCII/English so they render reliably with OpenCV's
Hershey fonts.  Pass ``labels`` to localize or use caller-provided copy.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    from .risk import BedRegion, RiskResult


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Joint:
    """A single normalized keypoint (0..1 in both axes)."""

    index: int
    x: float
    y: float
    confidence: float = 1.0


@dataclass(frozen=True)
class PersonSkeleton:
    """Skeleton for one detected person.

    Joints are normalized (0..1).  ``edges`` are pairs of joint indices; when
    omitted the standard COCO pose edges are used.
    """

    joints: Sequence[Joint]
    edges: Sequence[tuple[int, int]] | None = None
    track_id: int | None = None
    color: tuple[int, int, int] | None = None  # BGR override for this person


@dataclass(frozen=True)
class BedBoundary:
    """Normalized polygon defining the bed region."""

    points: Sequence[tuple[float, float]]
    label: str | None = None


@dataclass(frozen=True)
class RiskStatus:
    """Risk state produced by the risk lane, consumed for display only."""

    level: str  # "safe", "caution", or "danger"
    reasons: Sequence[str] = ()
    score: int | None = None


@dataclass(frozen=True)
class Telemetry:
    """Per-frame telemetry shown in the top-right panel."""

    fps: float | None = None
    device: str | None = None
    person_count: int = 0
    inference_ms: float | None = None


# ---------------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------------

# Standard COCO pose edges (0-indexed).
COCO_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)

# BGR palette used throughout the overlay.
PALETTE = {
    "safe": (76, 175, 80),
    "caution": (0, 165, 255),
    "danger": (54, 54, 255),
    "panel_bg": (28, 28, 32),
    "panel_border": (64, 64, 72),
    "text": (235, 235, 240),
    "text_muted": (160, 160, 170),
    "skeleton": (255, 205, 75),
    "bed": (128, 255, 160),
    "bed_glow": (60, 120, 72),
}

# Distinct fallback colors when multiple people are detected.
PERSON_COLORS: tuple[tuple[int, int, int], ...] = (
    (255, 205, 75),  # amber
    (255, 128, 128),  # coral
    (128, 200, 255),  # sky
    (200, 128, 255),  # lavender
    (128, 255, 180),  # mint
)

# Default ASCII/English labels.  Override via ``OverlayRenderer(..., labels=...)``.
DEFAULT_LABELS: dict[str, str] = {
    "safe": "SAFE",
    "caution": "CAUTION",
    "danger": "DANGER",
    "waiting": "WAITING",
    "fps": "FPS",
    "device": "DEVICE",
    "person_count": "PEOPLE",
    "inference": "INFERENCE",
    "bed": "BED",
    "reason": "REASONS",
    "no_person": "NONE",
    "window_title": "Fall Risk Monitor",
}

DEFAULT_REASON_TRANSLATIONS: dict[str, str] = {
    "body part outside bed": "Body part outside bed",
    "head or hip near bed edge": "Head or hip near bed edge",
    "upper body raised": "Upper body raised",
    "rapid movement toward edge": "Rapid movement toward edge",
}

# ASCII-safe calibration instructions (OpenCV Hershey fonts).
_CALIB_INSTRUCTIONS = (
    "CLICK 4 BED CORNERS",
    "ENTER/SPACE CONFIRM   R RESET   ESC CANCEL",
)


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


class OverlayRenderer:
    """Draw a complete monitoring overlay on incoming frames.

    The renderer never captures, infers, or evaluates risk.  It only paints
    what it is given and provides a thin Windows-compatible window helper.

    Example
    -------
    >>> renderer = OverlayRenderer()
    >>> annotated = renderer.render(
    ...     frame,
    ...     Telemetry(fps=29.7, device="CPU", person_count=1, inference_ms=42.0),
    ...     risk=RiskStatus("danger", reasons=["Body part outside bed"]),
    ...     persons=[PersonSkeleton(joints=...)],
    ...     bed_boundary=BedBoundary(points=[(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)]),
    ... )
    >>> renderer.show(annotated)
    >>> if renderer.should_quit():
    ...     renderer.close()
    """

    def __init__(
        self,
        window_name: str | None = None,
        *,
        labels: dict[str, str] | None = None,
        reason_translations: dict[str, str] | None = None,
        panel_alpha: float = 0.78,
        font_scale: float = 0.58,
        line_thickness: int = 2,
        skeleton_confidence_threshold: float = 0.35,
        show_bed_label: bool = True,
    ) -> None:
        self.window_name = window_name or DEFAULT_LABELS["window_title"]
        self.labels = {**DEFAULT_LABELS, **(labels or {})}
        self.reason_translations = {
            **DEFAULT_REASON_TRANSLATIONS,
            **(reason_translations or {}),
        }
        self.panel_alpha = panel_alpha
        self.font_scale = font_scale
        self.line_thickness = line_thickness
        self.skeleton_confidence_threshold = skeleton_confidence_threshold
        self.show_bed_label = show_bed_label
        self._window_created = False

    # -- Integration helpers ------------------------------------------------

    @classmethod
    def from_risk_result(
        cls,
        result: RiskResult,
        *,
        reason_translations: dict[str, str] | None = None,
    ) -> RiskStatus:
        """Convert a core ``RiskResult`` into a renderable ``RiskStatus``."""
        from .risk import RiskLevel

        level = {
            RiskLevel.SAFE: "safe",
            RiskLevel.CAUTION: "caution",
            RiskLevel.DANGER: "danger",
        }[result.level]
        translations = {**DEFAULT_REASON_TRANSLATIONS, **(reason_translations or {})}
        translated = tuple(translations.get(r, r) for r in result.reasons)
        return RiskStatus(level=level, reasons=translated, score=result.score)

    @classmethod
    def from_bed_region(cls, region: BedRegion) -> BedBoundary:
        """Convert a core ``BedRegion`` into a renderable polygon."""
        return BedBoundary(
            points=[
                (region.left, region.top),
                (region.right, region.top),
                (region.right, region.bottom),
                (region.left, region.bottom),
            ],
            label=DEFAULT_LABELS["bed"],
        )

    # -- Public rendering API -----------------------------------------------

    def render(
        self,
        frame: np.ndarray,
        telemetry: Telemetry,
        *,
        risk: RiskStatus | None = None,
        persons: Sequence[PersonSkeleton] = (),
        bed_boundary: BedBoundary | None = None,
    ) -> np.ndarray:
        """Return a new frame with the monitoring overlay drawn on top.

        The input ``frame`` is not modified.
        """
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a 3-channel BGR image")

        canvas = frame.copy()
        height, width = canvas.shape[:2]

        if bed_boundary is not None:
            self._draw_bed_boundary(canvas, bed_boundary, width, height)

        for idx, person in enumerate(persons):
            self._draw_person(canvas, person, width, height, person_index=idx)

        risk = risk or RiskStatus(level="waiting")
        self._draw_risk_panel(canvas, risk)
        self._draw_telemetry_panel(canvas, telemetry)

        return canvas

    def show(self, frame: np.ndarray) -> None:
        """Display the frame using ``cv2.imshow`` (Windows-compatible)."""
        cv2.imshow(self.window_name, frame)
        if not self._window_created:
            self._window_created = True

    def should_quit(self, timeout_ms: int = 1) -> bool:
        """Return True if the user pressed ESC or closed the window.

        ``timeout_ms`` is passed to ``cv2.waitKey``.  Use a small value inside
        a realtime loop.
        """
        key = cv2.waitKey(timeout_ms) & 0xFF
        if key == 27:  # ESC
            return True
        if self._window_created:
            try:
                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    return True
            except cv2.error:
                return True
        return False

    def close(self) -> None:
        """Destroy the OpenCV window if it was created."""
        if self._window_created:
            try:
                cv2.destroyWindow(self.window_name)
            except cv2.error:
                pass
            self._window_created = False

    def calibrate_bed(
        self,
        frame: np.ndarray,
        initial_region: BedBoundary | None = None,
        *,
        window_name: str | None = None,
    ) -> BedBoundary | None:
        """Run an interactive bed ROI calibration on a frozen camera frame.

        The frame is shown in a dedicated window.  The user left-clicks four
        corner points in the desired order to define the bed region.  Collected
        points and the in-progress polygon are overlaid on the frame, and
        ASCII-safe instructions are drawn at the bottom of the window.

        Keys
        ----
        Enter / Space
            Confirm the four-point selection and close the window.  This only
            works once four corners have been clicked.
        R / r
            Reset the selection to ``initial_region`` (or clear it if none).
        Esc
            Cancel and close the window.

        Parameters
        ----------
        frame
            3-channel BGR image to calibrate on.  It is not modified.
        initial_region
            Optional starting bed boundary in normalized coordinates.
        window_name
            Optional calibration window name.  Defaults to
            ``"{monitor_window} - Bed Calibration"``.

        Returns
        -------
        BedBoundary | None
            Normalized four-point boundary, or ``None`` if cancelled.
        """
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a 3-channel BGR image")

        height, width = frame.shape[:2]
        win = window_name or f"{self.window_name} - Bed Calibration"

        self._calib_cancelled = False
        self._calib_confirmed = False
        self._reset_calib_selection(frame.shape, initial_region)

        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.imshow(win, self._draw_calibration_overlay(frame, width, height))
        cv2.setMouseCallback(win, self._on_calib_mouse)

        try:
            while not self._calib_cancelled and not self._calib_confirmed:
                display = self._draw_calibration_overlay(frame, width, height)
                cv2.imshow(win, display)
                key = cv2.waitKey(30) & 0xFF
                if key == 27:  # Esc
                    self._calib_cancelled = True
                elif key in (13, 32):  # Enter / Space
                    if len(self._calib_points) == 4:
                        self._calib_confirmed = True
                elif key in (ord("r"), ord("R")):
                    self._reset_calib_selection(frame.shape, initial_region)
        finally:
            cv2.setMouseCallback(win, lambda *args, **kwargs: None)
            cv2.destroyWindow(win)

        if self._calib_cancelled or len(self._calib_points) < 4:
            return None

        points = [
            (max(0, min(width, x)) / width, max(0, min(height, y)) / height)
            for x, y in self._calib_points[:4]
        ]
        return BedBoundary(
            points=points,
            label=self.labels.get("bed", "BED"),
        )

    def calibrate_bed_live(
        self,
        read_frame: Callable[[], tuple[bool, np.ndarray]],
        initial_region: BedBoundary | None = None,
        *,
        window_name: str | None = None,
        timeout_ms: int = 33,
    ) -> BedBoundary | None:
        """Run interactive bed ROI calibration on a live camera feed.

        Frames are continuously read via ``read_frame`` and shown in a dedicated
        calibration window.  The user left-clicks four corner points in the
        desired order to define the bed region; collected points and the
        in-progress polygon are overlaid on each new frame.  Temporary read
        failures are handled gracefully: the last good frame is kept until a
        new one arrives.

        Keys
        ----
        Enter / Space
            Confirm the four-point selection and close the window.  This only
            works once four corners have been clicked.
        R / r
            Reset the selection to ``initial_region`` (or clear it if none).
        Esc
            Cancel and close the window.

        Parameters
        ----------
        read_frame
            Callable returning ``(success, frame)``.  ``frame`` must be a
            3-channel BGR image when ``success`` is ``True``.
        initial_region
            Optional starting bed boundary in normalized coordinates.
        window_name
            Optional calibration window name.  Defaults to
            ``"{monitor_window} - Bed Calibration"``.
        timeout_ms
            Maximum time to wait for a key each frame (default 33 ms, ~30 FPS).
            This prevents busy looping.

        Returns
        -------
        BedBoundary | None
            Normalized four-point boundary confirmed by the user, or ``None``
            if cancelled or no valid frame was ever available to normalize
            against.  The caller can retain the latest valid frame by calling
            ``read_frame`` after calibration returns.
        """
        if not callable(read_frame):
            raise TypeError("read_frame must be callable")

        win = window_name or f"{self.window_name} - Bed Calibration"
        self._calib_cancelled = False
        self._calib_confirmed = False
        self._calib_current_frame: np.ndarray | None = None

        fallback = np.zeros((480, 640, 3), dtype=np.uint8)
        have_valid_frame = False
        last_valid_shape: tuple[int, ...] | None = None

        # Initialize selection using fallback dimensions so the overlay can be
        # drawn before the first live frame arrives.  It is rescaled when the
        # first valid frame is read.
        self._reset_calib_selection(fallback.shape, initial_region)

        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win, self._on_calib_mouse)

        try:
            while not self._calib_cancelled and not self._calib_confirmed:
                success, frame = read_frame()
                if (
                    success
                    and frame is not None
                    and frame.ndim == 3
                    and frame.shape[2] == 3
                ):
                    self._calib_current_frame = frame.copy()
                    last_valid_shape = self._calib_current_frame.shape
                    if not have_valid_frame:
                        have_valid_frame = True
                        self._reset_calib_selection(
                            self._calib_current_frame.shape, initial_region
                        )
                elif self._calib_current_frame is None:
                    self._calib_current_frame = fallback

                display_frame = self._calib_current_frame
                assert display_frame is not None
                height, width = display_frame.shape[:2]
                display = self._draw_calibration_overlay(
                    display_frame, width, height
                )
                cv2.imshow(win, display)

                key = cv2.waitKey(timeout_ms) & 0xFF
                if key == 27:  # Esc
                    self._calib_cancelled = True
                elif key in (13, 32):  # Enter / Space
                    if len(self._calib_points) == 4 and have_valid_frame:
                        self._calib_confirmed = True
                elif key in (ord("r"), ord("R")):
                    if have_valid_frame and self._calib_current_frame is not None:
                        self._reset_calib_selection(
                            self._calib_current_frame.shape, initial_region
                        )
                    else:
                        self._reset_calib_selection(fallback.shape, initial_region)
        finally:
            cv2.setMouseCallback(win, lambda *args, **kwargs: None)
            cv2.destroyWindow(win)
            self._calib_current_frame = None

        if (
            self._calib_cancelled
            or len(self._calib_points) < 4
            or not have_valid_frame
            or last_valid_shape is None
        ):
            return None

        height, width = last_valid_shape[:2]
        points = [
            (max(0, min(width, x)) / width, max(0, min(height, y)) / height)
            for x, y in self._calib_points[:4]
        ]
        return BedBoundary(
            points=points,
            label=self.labels.get("bed", "BED"),
        )

    # -- Drawing internals --------------------------------------------------

    def _risk_color(self, level: str) -> tuple[int, int, int]:
        if level == "danger":
            return PALETTE["danger"]
        if level == "caution":
            return PALETTE["caution"]
        return PALETTE["safe"]

    def _risk_label(self, level: str) -> str:
        return self.labels.get(level, level.upper())

    def _draw_risk_panel(self, canvas: np.ndarray, risk: RiskStatus) -> None:
        _, w = canvas.shape[:2]
        margin = max(12, int(w * 0.018))
        panel_w = min(360, int(w * 0.42))
        accent = 8
        radius = 10

        status = self._risk_label(risk.level)
        color = self._risk_color(risk.level)

        # Measure content first.
        status_font = self.font_scale * 2.2
        (_, status_h), _ = cv2.getTextSize(
            status, cv2.FONT_HERSHEY_SIMPLEX, status_font, 2
        )

        reason_lines: list[str] = []
        if risk.reasons:
            reason_font = self.font_scale * 0.92
            for reason in risk.reasons[:3]:
                line = f"- {reason}"
                (rw, _), _ = cv2.getTextSize(
                    line, cv2.FONT_HERSHEY_SIMPLEX, reason_font, 1
                )
                if rw > panel_w - margin * 2 - accent:
                    # crude truncation: keep clipping text visually simple
                    line = (
                        line[
                            : int(
                                len(line) * (panel_w - margin * 2 - accent) / max(1, rw)
                            )
                        ]
                        + "..."
                    )
                reason_lines.append(line)

        line_height = int(self.font_scale * 28)
        panel_h = margin * 2 + status_h + 10
        if risk.score is not None:
            panel_h += line_height
        if reason_lines:
            panel_h += (len(reason_lines) + 1) * line_height + 8

        x1, y1 = margin, margin
        x2, y2 = x1 + panel_w, y1 + panel_h

        # Background panel with rounded corners.
        self._fill_round_rect(canvas, (x1, y1), (x2, y2), PALETTE["panel_bg"], radius)
        cv2.rectangle(
            canvas, (x1, y1), (x2, y2), PALETTE["panel_border"], 1, cv2.LINE_AA
        )

        # Left accent bar.
        bar_x1 = x1 + margin
        bar_y1 = y1 + margin
        bar_y2 = y2 - margin
        cv2.rectangle(
            canvas, (bar_x1, bar_y1), (bar_x1 + accent, bar_y2), color, -1, cv2.LINE_AA
        )
        # Soft glow behind the bar.
        glow = canvas.copy()
        cv2.rectangle(
            glow,
            (bar_x1 - 4, bar_y1),
            (bar_x1 + accent + 6, bar_y2),
            color,
            -1,
            cv2.LINE_AA,
        )
        cv2.addWeighted(glow, 0.18, canvas, 0.82, 0, canvas)

        text_x = bar_x1 + accent + margin
        text_y = y1 + margin + status_h
        cv2.putText(
            canvas,
            status,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            status_font,
            color,
            2,
            cv2.LINE_AA,
        )

        cursor_y = text_y + line_height
        if risk.score is not None:
            score_text = f"score: {risk.score}"
            cv2.putText(
                canvas,
                score_text,
                (text_x, cursor_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale * 0.85,
                PALETTE["text_muted"],
                1,
                cv2.LINE_AA,
            )
            cursor_y += line_height

        if reason_lines:
            cursor_y += 4
            reason_font = self.font_scale * 0.92
            cv2.putText(
                canvas,
                self.labels["reason"],
                (text_x, cursor_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                reason_font,
                PALETTE["text_muted"],
                1,
                cv2.LINE_AA,
            )
            cursor_y += line_height
            for line in reason_lines:
                cv2.putText(
                    canvas,
                    line,
                    (text_x, cursor_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    reason_font,
                    PALETTE["text"],
                    1,
                    cv2.LINE_AA,
                )
                cursor_y += line_height

    def _draw_telemetry_panel(self, canvas: np.ndarray, telemetry: Telemetry) -> None:
        _, w = canvas.shape[:2]
        margin = max(12, int(w * 0.018))
        panel_w = min(260, int(w * 0.30))
        radius = 10
        line_height = int(self.font_scale * 28)

        fps = f"{telemetry.fps:.1f}" if telemetry.fps is not None else "--"
        device = telemetry.device or "--"
        count = (
            str(telemetry.person_count)
            if telemetry.person_count
            else self.labels["no_person"]
        )
        inference = (
            f"{telemetry.inference_ms:.1f} ms"
            if telemetry.inference_ms is not None
            else "--"
        )

        lines = [
            f"{self.labels['fps']}: {fps}",
            f"{self.labels['device']}: {device}",
            f"{self.labels['person_count']}: {count}",
            f"{self.labels['inference']}: {inference}",
        ]

        (line_w, _), _ = cv2.getTextSize(
            max(lines, key=len), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1
        )
        panel_w = max(panel_w, line_w + margin * 3)
        panel_h = margin * 2 + len(lines) * line_height

        x2, y2 = w - margin, margin + panel_h
        x1, y1 = x2 - panel_w, margin

        self._fill_round_rect(canvas, (x1, y1), (x2, y2), PALETTE["panel_bg"], radius)
        cv2.rectangle(
            canvas, (x1, y1), (x2, y2), PALETTE["panel_border"], 1, cv2.LINE_AA
        )

        y = y1 + margin + int(self.font_scale * 20)
        for line in lines:
            cv2.putText(
                canvas,
                line,
                (x1 + margin, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                PALETTE["text"],
                1,
                cv2.LINE_AA,
            )
            y += line_height

    def _draw_bed_boundary(
        self,
        canvas: np.ndarray,
        boundary: BedBoundary,
        width: int,
        height: int,
    ) -> None:
        valid_points = [
            (p[0], p[1])
            for p in boundary.points
            if math.isfinite(p[0]) and math.isfinite(p[1])
        ]
        if len(valid_points) < 2:
            return

        pts = np.array(
            [[int(p[0] * width), int(p[1] * height)] for p in valid_points],
            dtype=np.int32,
        )
        pts = np.clip(pts, 0, [width - 1, height - 1])

        # Closed polygon with dashed edges.
        closed = np.concatenate([pts, pts[:1]])
        for i in range(len(closed) - 1):
            p1 = tuple(closed[i])
            p2 = tuple(closed[i + 1])
            self._draw_dashed_line(canvas, p1, p2, PALETTE["bed"], 2, dash_len=12)

        # Soft interior fill.
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [pts], PALETTE["bed_glow"])
        cv2.addWeighted(overlay, 0.12, canvas, 0.88, 0, canvas)

        if self.show_bed_label and boundary.label:
            cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
            self._draw_centered_text(
                canvas,
                boundary.label,
                (cx, cy),
                self.font_scale * 0.9,
                PALETTE["bed"],
                PALETTE["panel_bg"],
            )

    def _draw_person(
        self,
        canvas: np.ndarray,
        person: PersonSkeleton,
        width: int,
        height: int,
        person_index: int = 0,
    ) -> None:
        joints = {j.index: j for j in person.joints}
        if not joints:
            return

        edges = person.edges if person.edges is not None else COCO_EDGES
        color = person.color or PERSON_COLORS[person_index % len(PERSON_COLORS)]

        for a, b in edges:
            ja = joints.get(a)
            jb = joints.get(b)
            if ja is None or jb is None:
                continue
            if (
                ja.confidence < self.skeleton_confidence_threshold
                or jb.confidence < self.skeleton_confidence_threshold
            ):
                continue
            pa = self._norm_to_pixel(ja.x, ja.y, width, height)
            pb = self._norm_to_pixel(jb.x, jb.y, width, height)
            if pa is None or pb is None:
                continue
            cv2.line(canvas, pa, pb, color, self.line_thickness, cv2.LINE_AA)

        for j in joints.values():
            if j.confidence < self.skeleton_confidence_threshold:
                continue
            p = self._norm_to_pixel(j.x, j.y, width, height)
            if p is None:
                continue
            radius = 4 if j.confidence >= 0.7 else 3
            cv2.circle(canvas, p, radius, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, p, radius + 1, (255, 255, 255), 1, cv2.LINE_AA)

    # -- Low-level helpers --------------------------------------------------

    def _norm_to_pixel(
        self, x: float, y: float, width: int, height: int
    ) -> tuple[int, int] | None:
        """Convert normalized coordinates to pixel space.

        Returns ``None`` for non-finite values so they are never passed to
        ``int()`` or OpenCV drawing functions.
        """
        if not (math.isfinite(x) and math.isfinite(y)):
            return None
        return (int(x * width), int(y * height))

    def _fill_round_rect(
        self,
        img: np.ndarray,
        top_left: tuple[int, int],
        bottom_right: tuple[int, int],
        color: tuple[int, int, int],
        radius: int,
    ) -> None:
        x1, y1 = top_left
        x2, y2 = bottom_right
        r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)

        overlay = img.copy()
        cv2.rectangle(overlay, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(overlay, (x1, y1 + r), (x2, y2 - r), color, -1)
        cv2.circle(overlay, (x1 + r, y1 + r), r, color, -1)
        cv2.circle(overlay, (x2 - r, y1 + r), r, color, -1)
        cv2.circle(overlay, (x1 + r, y2 - r), r, color, -1)
        cv2.circle(overlay, (x2 - r, y2 - r), r, color, -1)
        cv2.addWeighted(overlay, self.panel_alpha, img, 1 - self.panel_alpha, 0, img)

    def _draw_dashed_line(
        self,
        img: np.ndarray,
        p1: tuple[int, int],
        p2: tuple[int, int],
        color: tuple[int, int, int],
        thickness: int,
        dash_len: int = 10,
    ) -> None:
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        length = float(np.hypot(dx, dy))
        if length == 0:
            return
        steps = max(1, int(length / dash_len))
        for i in range(steps):
            if i % 2 == 1:
                continue
            t0 = i / steps
            t1 = min(1.0, (i + 1) / steps)
            a = (int(p1[0] + dx * t0), int(p1[1] + dy * t0))
            b = (int(p1[0] + dx * t1), int(p1[1] + dy * t1))
            cv2.line(img, a, b, color, thickness, cv2.LINE_AA)

    def _draw_centered_text(
        self,
        img: np.ndarray,
        text: str,
        center: tuple[int, int],
        font_scale: float,
        color: tuple[int, int, int],
        bg_color: tuple[int, int, int],
    ) -> None:
        (tw, th), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        x1 = center[0] - tw // 2 - 6
        y1 = center[1] - th // 2 - 4
        x2 = center[0] + tw // 2 + 6
        y2 = center[1] + th // 2 + baseline + 4
        cv2.rectangle(img, (x1, y1), (x2, y2), bg_color, -1, cv2.LINE_AA)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        cv2.putText(
            img,
            text,
            (x1 + 6, y2 - baseline - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            1,
            cv2.LINE_AA,
        )

    # -- Calibration helpers -------------------------------------------------

    def _on_calib_mouse(
        self, event: int, x: int, y: int, flags: int, param: Any
    ) -> None:
        """Mouse callback for the bed calibration window.

        Each left click appends a corner point.  A fifth click (or any click
        after four points have been collected) starts a fresh selection so the
        user can quickly redo the polygon.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self._calib_points) >= 4:
                self._calib_points = [(x, y)]
            else:
                self._calib_points.append((x, y))

    def _reset_calib_selection(
        self,
        frame_shape: tuple[int, ...],
        initial_region: BedBoundary | None,
    ) -> None:
        """Reset the calibration selection to ``initial_region`` or clear it."""
        height, width = frame_shape[:2]
        if initial_region is not None:
            self._calib_points = [
                (int(p[0] * width), int(p[1] * height))
                for p in initial_region.points
                if math.isfinite(p[0]) and math.isfinite(p[1])
            ]
        else:
            self._calib_points = []

    def _draw_calibration_overlay(
        self, frame: np.ndarray, width: int, height: int
    ) -> np.ndarray:
        """Return a copy of ``frame`` with the current points and instructions."""
        display = frame.copy()

        points = self._calib_points
        if points:
            pts = np.array(points, dtype=np.int32)
            pts = np.clip(pts, 0, [width - 1, height - 1])

            # Draw collected corner points.
            for x, y in pts:
                cv2.circle(
                    display, (int(x), int(y)), 6, PALETTE["bed"], -1, cv2.LINE_AA
                )
                cv2.circle(
                    display, (int(x), int(y)), 6, (255, 255, 255), 1, cv2.LINE_AA
                )

            # Draw polygon preview in click order.
            if len(pts) >= 2:
                for i in range(len(pts) - 1):
                    self._draw_dashed_line(
                        display,
                        tuple(pts[i]),
                        tuple(pts[i + 1]),
                        PALETTE["bed"],
                        2,
                        dash_len=12,
                    )

            if len(pts) == 4:
                # Close the polygon and shade the interior once four corners
                # have been collected.
                self._draw_dashed_line(
                    display,
                    tuple(pts[-1]),
                    tuple(pts[0]),
                    PALETTE["bed"],
                    2,
                    dash_len=12,
                )
                overlay = display.copy()
                cv2.fillPoly(overlay, [pts], PALETTE["bed_glow"])
                cv2.addWeighted(
                    overlay, self.panel_alpha, display, 1 - self.panel_alpha, 0, display
                )

        self._draw_instruction_strip(display)
        return display

    def _draw_instruction_strip(self, display: np.ndarray) -> None:
        """Draw the calibration instruction strip at the bottom of the frame."""
        height, width = display.shape[:2]
        margin = max(10, int(width * 0.015))
        line_height = int(self.font_scale * 28)
        panel_h = margin * 2 + len(_CALIB_INSTRUCTIONS) * line_height

        overlay = display.copy()
        cv2.rectangle(overlay, (0, height - panel_h), (width, height), PALETTE["panel_bg"], -1)
        cv2.addWeighted(
            overlay, self.panel_alpha, display, 1 - self.panel_alpha, 0, display
        )

        y = height - panel_h + margin + int(self.font_scale * 20)
        for line in _CALIB_INSTRUCTIONS:
            cv2.putText(
                display,
                line,
                (margin, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                PALETTE["text"],
                1,
                cv2.LINE_AA,
            )
            y += line_height
