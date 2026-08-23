import cv2
import numpy as np
import pytest

from falling_prediction.homography import UNIT_SQUARE, compute_homography, transform_keypoints


def test_unit_square_corners_map_to_themselves():
    homography = compute_homography(UNIT_SQUARE)

    mapped = cv2.perspectiveTransform(UNIT_SQUARE.reshape(-1, 1, 2), homography).reshape(-1, 2)

    np.testing.assert_allclose(mapped, UNIT_SQUARE, rtol=0, atol=1e-6)


def test_trapezoid_corners_map_to_corresponding_unit_square_corners():
    trapezoid = np.array(
        ((0.20, 0.20), (0.80, 0.10), (0.90, 0.80), (0.10, 0.90)),
        dtype=np.float32,
    )
    homography = compute_homography(trapezoid)

    mapped = cv2.perspectiveTransform(trapezoid.reshape(-1, 1, 2), homography).reshape(-1, 2)

    np.testing.assert_allclose(mapped, UNIT_SQUARE, rtol=0, atol=1e-5)


def test_transform_keypoints_preserves_confidence_and_invalid_nan_rows():
    keypoints = np.full((17, 3), np.nan, dtype=float)
    keypoints[0] = (0.10, 0.20, 0.13)
    keypoints[1] = (0.40, 0.50, 0.87)
    keypoints[2] = (np.nan, 0.30, 0.42)
    keypoints[3] = (0.60, np.nan, 0.99)
    homography = np.array(((2.0, 0.0, 0.1), (0.0, 3.0, 0.2), (0.0, 0.0, 1.0)))

    transformed = transform_keypoints(keypoints, homography)

    np.testing.assert_allclose(transformed[:2, :2], ((0.3, 0.8), (0.9, 1.7)))
    np.testing.assert_array_equal(transformed[:, 2], keypoints[:, 2])
    invalid = keypoints[2:]
    np.testing.assert_array_equal(np.isnan(transformed[2:]), np.isnan(invalid))
    np.testing.assert_array_equal(transformed[2:][~np.isnan(invalid)], invalid[~np.isnan(invalid)])


@pytest.mark.parametrize(
    "points",
    [
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)),
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (np.nan, 1.0)),
        ((-0.1, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        ((0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0)),
    ],
)
def test_compute_homography_rejects_invalid_points(points):
    with pytest.raises(ValueError):
        compute_homography(points)


@pytest.mark.parametrize(
    "keypoints, homography",
    [
        (np.zeros((16, 3)), np.eye(3)),
        (np.zeros((17, 2)), np.eye(3)),
        (np.zeros((17, 3)), np.zeros((2, 2))),
        (np.zeros((17, 3)), np.full((3, 3), np.nan)),
    ],
)
def test_transform_keypoints_rejects_invalid_input(keypoints, homography):
    with pytest.raises(ValueError):
        transform_keypoints(keypoints, homography)
