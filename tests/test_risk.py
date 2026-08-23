import numpy as np

import pytest

from falling_prediction.risk import BedRegion, RiskEvaluator, RiskLevel


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


def test_movement_toward_edge_is_temporal_signal():
    evaluator = RiskEvaluator()
    evaluator.evaluate(pose(0.50))
    result = evaluator.evaluate(pose(0.78))
    assert result.score == 2
    assert result.level is RiskLevel.CAUTION
    assert "movement toward edge" in result.reasons[-1]


def test_polygon_containment_and_edge_distance_are_used():
    bed = BedRegion(points=((0.2, 0.2), (0.8, 0.3), (0.7, 0.8), (0.25, 0.7)))
    assert bed.contains((0.5, 0.5))
    assert not bed.contains((0.05, 0.5))
    assert bed.distance_to_edges((0.2, 0.2)) == pytest.approx(0)


def test_bed_polygon_requires_exactly_four_valid_points():
    with pytest.raises(ValueError):
        BedRegion(points=((0, 0), (1, 0), (1, 1)))
    with pytest.raises(ValueError):
        BedRegion(points=((0, 0), (1, 1), (0, 1), (1, 0)))
