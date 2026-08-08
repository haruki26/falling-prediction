"""OpenVINO contract and inference adapter for human-pose-estimation-0001."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

INPUT_SHAPE = (1, 3, 256, 456)


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
        outputs = list(self.model.outputs)
        found: dict[str, Any] = {}
        for output in outputs:
            dims = tuple(int(x) for x in output.shape)
            _channels = dims[1] if len(dims) == 4 else -1
            if dims == (1, 38, 32, 57):
                found["paf"] = output
            elif dims == (1, 19, 32, 57):
                found["heatmap"] = output
        if set(found) != {"paf", "heatmap"}:
            raise ValueError(
                "model must expose PAF [1,38,32,57] and heatmap [1,19,32,57]"
            )
        self.paf_output, self.heatmap_output = found["paf"], found["heatmap"]
        self.compiled_model = core.compile_model(self.model, self.device)

    def infer(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be HWC BGR")
        tensor = (
            cv2.resize(image, (456, 256), interpolation=cv2.INTER_LINEAR)
            .transpose(2, 0, 1)[None]
            .astype(np.float32)
        )
        result = self.compiled_model([tensor])
        return np.asarray(result[self.paf_output]), np.asarray(
            result[self.heatmap_output]
        )
