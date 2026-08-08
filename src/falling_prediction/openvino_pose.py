"""OpenVINO contract and inference adapter for human-pose-estimation-0001."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

INPUT_SHAPE = (1, 3, 256, 456)
PAF_NAME = "Mconv7_stage2_L1"
HEATMAP_NAME = "Mconv7_stage2_L2"
PAF_SHAPE = (1, 38, 32, 57)
HEATMAP_SHAPE = (1, 19, 32, 57)


def available_devices() -> tuple[str, ...]:
    from openvino import Core

    return tuple(Core().available_devices)


class PoseEstimator:
    def __init__(self, model_path: str | Path, device: str = "CPU") -> None:
        from openvino import Core

        self.model_path, self.device = Path(model_path), device.upper()
        if self.model_path.suffix.lower() != ".xml" or not self.model_path.is_file():
            raise FileNotFoundError(f"OpenVINO IR XML not found: {self.model_path}")
        if not self.model_path.with_suffix(".bin").is_file():
            raise FileNotFoundError(
                f"OpenVINO IR BIN not found: {self.model_path.with_suffix('.bin')}"
            )
        core = Core()
        devices = {d.upper().split(".")[0] for d in core.available_devices}
        if self.device not in devices:
            raise ValueError(
                f"device {self.device} unavailable; available: {sorted(devices)}"
            )
        self.model = core.read_model(self.model_path)
        shape = tuple(int(x) for x in self.model.input(0).shape)
        if shape != INPUT_SHAPE:
            raise ValueError(
                f"human-pose-estimation-0001 input must be {INPUT_SHAPE}, got {shape}"
            )
        required = {PAF_NAME: PAF_SHAPE, HEATMAP_NAME: HEATMAP_SHAPE}
        found: dict[str, Any] = {}
        for output in self.model.outputs:
            names = {str(name) for name in output.get_names()}
            for name, expected_shape in required.items():
                if name in names:
                    shape = tuple(int(x) for x in output.shape)
                    if shape != expected_shape:
                        raise ValueError(
                            f"output {name} must have shape {expected_shape}, got {shape}"
                        )
                    found[name] = output
        if set(found) != set(required):
            raise ValueError(
                "model must expose exact outputs "
                f"{PAF_NAME} {PAF_SHAPE} and {HEATMAP_NAME} {HEATMAP_SHAPE}"
            )
        self.compiled_model = core.compile_model(self.model, self.device)
        self._request = self.compiled_model.create_infer_request()

        # Keep integer indexes only.  A model Output is not a valid key for
        # the result map on all OpenVINO versions (notably Windows builds).
        compiled_indexes: dict[str, int] = {}
        for index, output in enumerate(self.compiled_model.outputs):
            names = {str(name) for name in output.get_names()}
            for name in required:
                if name in names:
                    compiled_indexes[name] = index
        if set(compiled_indexes) != set(required):
            raise ValueError("compiled model lost one or more required named outputs")
        self._paf_index = compiled_indexes[PAF_NAME]
        self._heatmap_index = compiled_indexes[HEATMAP_NAME]
        self._input_name = self.model.input(0).get_any_name()

    def infer(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be HWC BGR")
        tensor = (
            cv2.resize(image, (456, 256), interpolation=cv2.INTER_LINEAR)
            .transpose(2, 0, 1)[None]
            .astype(np.float32)
        )
        self._request.infer({self._input_name: tensor})
        paf = np.asarray(self._request.get_output_tensor(self._paf_index).data)
        heatmap = np.asarray(self._request.get_output_tensor(self._heatmap_index).data)
        return paf, heatmap
