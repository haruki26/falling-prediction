import json

import numpy as np
import pytest

from falling_prediction.calibration import PerspectiveCalibration, load_calibration, save_calibration
from falling_prediction.config import ConfigurationError


def calibration():
    return PerspectiveCalibration(((.1, .2), (.9, .1), (.85, .8), (.15, .9)), 100, 80)


def test_v2_round_trip_records_perspective_metadata(tmp_path):
    path = tmp_path / "calibration.json"
    save_calibration(path, calibration())
    loaded = load_calibration(path)
    assert loaded == calibration()
    assert json.loads(path.read_text())["version"] == 2


def test_v1_is_not_reinterpreted_or_overwritten(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"version": 1, "left": .1, "top": .1, "right": .9, "bottom": .9}))
    with pytest.raises(ConfigurationError, match="recalibration"):
        load_calibration(path)
    assert path.exists()


def test_rectification_produces_canonical_bgr_frame():
    source = np.zeros((80, 100, 3), dtype=np.uint8)
    source[:, :, 0] = np.arange(100, dtype=np.uint8)
    corrected = calibration().rectify(source)
    assert corrected.shape == (256, 456, 3)
    assert corrected.dtype == source.dtype
