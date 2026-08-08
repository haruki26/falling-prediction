import numpy as np
import pytest

from falling_prediction.pose_decoder import _convert_openpose_to_coco, decode_poses


def test_openpose_to_coco_conversion_uses_canonical_mapping():
    openpose = np.arange(18 * 3, dtype=float).reshape(18, 3)
    expected_indices = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]

    converted = _convert_openpose_to_coco(openpose)

    np.testing.assert_array_equal(converted, openpose[expected_indices])


def test_decoder_empty_output_has_canonical_shapes():
    poses, scores = decode_poses(np.zeros((1, 38, 32, 57)), np.zeros((1, 19, 32, 57)))
    assert poses.shape == (0, 17, 3)
    assert scores.shape == (0,)


def test_decoder_rejects_wrong_topology():
    with pytest.raises(ValueError, match="19"):
        decode_poses(np.zeros((1, 38, 32, 57)), np.zeros((1, 17, 32, 57)))

    with pytest.raises(ValueError, match="38"):
        decode_poses(np.zeros((1, 36, 32, 57)), np.zeros((1, 19, 32, 57)))
