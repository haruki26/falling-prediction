from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np


class _FakeCv2(ModuleType):
    INTER_LINEAR = 1

    @staticmethod
    def resize(image, size, interpolation=1):
        return np.zeros((size[1], size[0], 3), dtype=image.dtype)


class _Output:
    def __init__(self, name, shape):
        self._name, self.shape = name, shape

    def get_names(self):
        return {self._name}

    def get_any_name(self):
        return self._name


class _Request:
    def __init__(self):
        self.inputs = None

    def infer(self, inputs):
        self.inputs = inputs

    def get_output_tensor(self, index):
        return SimpleNamespace(data=np.full((1, 38 if index == 0 else 19, 32, 57), index, dtype=np.float32))


class _Compiled:
    outputs = [_Output("Mconv7_stage2_L1", (1, 38, 32, 57)), _Output("Mconv7_stage2_L2", (1, 19, 32, 57))]

    def create_infer_request(self):
        return _Request()

    def __call__(self, *args, **kwargs):
        raise AssertionError("compiled result maps must not be used")


class _Model:
    outputs = _Compiled.outputs

    def input(self, index):
        return _Output("image", (1, 3, 256, 456))


class _Core:
    available_devices = ("CPU",)

    def read_model(self, path):
        return _Model()

    def compile_model(self, model, device):
        return _Compiled()


def test_infer_uses_request_tensor_indexes_not_output_keys(tmp_path, monkeypatch):
    xml = tmp_path / "pose.xml"
    xml.write_text("xml")
    xml.with_suffix(".bin").write_bytes(b"bin")
    monkeypatch.setitem(sys.modules, "cv2", _FakeCv2("cv2"))
    monkeypatch.setitem(sys.modules, "openvino", SimpleNamespace(Core=_Core))

    # Import after faking optional native dependencies.
    sys.modules.pop("falling_prediction.openvino_pose", None)
    from falling_prediction.openvino_pose import PoseEstimator

    estimator = PoseEstimator(xml)
    paf, heatmap = estimator.infer(np.zeros((480, 640, 3), dtype=np.uint8))
    assert paf.shape == (1, 38, 32, 57)
    assert heatmap.shape == (1, 19, 32, 57)
    assert not hasattr(estimator, "paf_output")
