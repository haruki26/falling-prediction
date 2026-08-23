import numpy as np

from falling_prediction.risk import (
    MIN_TORSO_LENGTH,
    UPPER_BODY_RAISED_MAX_ANGLE_DEGREES,
    RiskEvaluator,
    RiskLevel,
)


def pose(x=0.5, y=0.5):
    points = np.zeros((17, 3), dtype=float)
    points[:, :2] = (x, y)
    points[:, 2] = 1.0
    return points


def test_centered_pose_is_safe():
    result = RiskEvaluator().evaluate(pose())
    assert result.level is RiskLevel.SAFE
    assert result.reasons == ()


def test_body_outside_bed_is_danger():
    points = pose()
    points[15, :2] = (0.95, 0.5)
    result = RiskEvaluator().evaluate(points)
    assert result.level is RiskLevel.DANGER
    assert "outside bed" in result.reasons[0]


def test_edge_and_raised_upper_body_is_danger():
    points = pose(0.82, 0.5)
    points[[5, 6], 1] = 0.30
    points[[11, 12], 1] = 0.55
    result = RiskEvaluator().evaluate(points)
    assert result.level is RiskLevel.DANGER


def set_torso(points, shoulder_midpoint, hip_midpoint):
    sx, sy = shoulder_midpoint
    hx, hy = hip_midpoint
    points[5, :2] = (sx - 0.04, sy)
    points[6, :2] = (sx + 0.04, sy)
    points[11, :2] = (hx - 0.04, hy)
    points[12, :2] = (hx + 0.04, hy)
    return points


def test_near_vertical_torso_triggers_upper_body_raised():
    points = set_torso(pose(), (0.5, 0.32), (0.5, 0.62))
    result = RiskEvaluator().evaluate(points)
    assert result.score == 1
    assert result.reasons == ("upper body raised",)


def test_near_horizontal_torso_does_not_trigger_upper_body_raised():
    points = set_torso(pose(), (0.38, 0.5), (0.62, 0.52))
    result = RiskEvaluator().evaluate(points)
    assert "upper body raised" not in result.reasons
    assert result.score == 0


def test_torso_angle_threshold_is_inclusive():
    # atan2(horizontal, vertical) is the image-plane angle from vertical.
    angle = np.deg2rad(UPPER_BODY_RAISED_MAX_ANGLE_DEGREES)
    points = set_torso(pose(), (0.5, 0.35),
                       (0.5 + 0.25 * np.sin(angle), 0.35 + 0.25 * np.cos(angle)))
    result = RiskEvaluator().evaluate(points)
    assert "upper body raised" in result.reasons

    just_over = angle + np.deg2rad(1.0)
    points = set_torso(pose(), (0.5, 0.35),
                       (0.5 + 0.25 * np.sin(just_over), 0.35 + 0.25 * np.cos(just_over)))
    result = RiskEvaluator().evaluate(points)
    assert "upper body raised" not in result.reasons


def test_insufficient_or_invalid_torso_keypoints_do_not_trigger():
    points = set_torso(pose(), (0.5, 0.5), (0.5, 0.5 + MIN_TORSO_LENGTH / 2))
    assert "upper body raised" not in RiskEvaluator().evaluate(points).reasons

    points = set_torso(pose(), (0.5, 0.35), (0.5, 0.65))
    points[6, 2] = 0.0
    assert "upper body raised" not in RiskEvaluator().evaluate(points).reasons

    points = set_torso(pose(), (0.5, 0.35), (0.5, 0.65))
    points[11, 0] = np.nan
    assert "upper body raised" not in RiskEvaluator().evaluate(points).reasons


def test_movement_toward_edge_is_temporal_signal():
    evaluator = RiskEvaluator()
    evaluator.evaluate(pose(0.50))
    result = evaluator.evaluate(pose(0.78))
    assert result.score == 2
    assert result.level is RiskLevel.CAUTION
    assert "movement toward edge" in result.reasons[-1]
