"""Windows-friendly, explicit runtime configuration."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path


class ConfigurationError(ValueError):
    """Raised when a requested runtime resource is not usable."""


@dataclass(frozen=True)
class AppConfig:
    camera_index: int = 0
    model_path: Path | None = None
    device: str = "CPU"
    bed_left: float = 0.10
    bed_top: float = 0.10
    bed_right: float = 0.90
    bed_bottom: float = 0.90


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bed fall-risk MVP")
    parser.add_argument(
        "--camera",
        "--camera-index",
        dest="camera_index",
        type=int,
        default=0,
        help="native camera index (default: 0)",
    )
    parser.add_argument(
        "--model",
        "--model-path",
        dest="model_path",
        type=Path,
        required=True,
        help="path to the human-pose-estimation-0001 .xml IR model",
    )
    parser.add_argument(
        "--device",
        default="CPU",
        type=str.upper,
        help="OpenVINO device, exactly CPU, GPU, or NPU (default: CPU)",
    )
    for name in ("left", "top", "right", "bottom"):
        parser.add_argument(
            f"--bed-{name}",
            dest=f"bed_{name}",
            type=float,
            help=f"normalized bed rectangle {name} (default from 0.10/0.90)",
        )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> AppConfig:
    args = build_parser().parse_args(argv)
    return AppConfig(
        args.camera_index,
        args.model_path,
        args.device,
        args.bed_left if args.bed_left is not None else 0.10,
        args.bed_top if args.bed_top is not None else 0.10,
        args.bed_right if args.bed_right is not None else 0.90,
        args.bed_bottom if args.bed_bottom is not None else 0.90,
    )


def validate_config(
    config: AppConfig,
    available_devices: Sequence[str] | None = None,
    camera_probe: Callable[[int], bool] | None = None,
) -> AppConfig:
    """Validate all requested resources; never silently substitutes a device.

    ``camera_probe`` is injectable so CLI validation does not require OpenCV and tests
    remain portable.  Actual capture code can pass a probe before starting capture.
    """
    if config.camera_index < 0:
        raise ConfigurationError("camera index must be >= 0")
    if config.device not in {"CPU", "GPU", "NPU"}:
        raise ConfigurationError("device must be one of CPU, GPU, or NPU")
    if not (
        0 <= config.bed_left < config.bed_right <= 1
        and 0 <= config.bed_top < config.bed_bottom <= 1
    ):
        raise ConfigurationError(
            "bed rectangle must satisfy 0 <= left < right <= 1 and 0 <= top < bottom <= 1"
        )
    if config.model_path is None:
        raise ConfigurationError("model path is required")
    if config.model_path.suffix.lower() != ".xml":
        raise ConfigurationError("model path must point to an OpenVINO .xml IR file")
    if not config.model_path.is_file():
        raise ConfigurationError(f"model XML does not exist: {config.model_path}")
    if config.model_path.with_suffix(".bin").is_file() is False:
        raise ConfigurationError(
            f"model BIN sibling does not exist: {config.model_path.with_suffix('.bin')}"
        )
    if available_devices is None:
        from .openvino_pose import available_devices as get_devices

        available_devices = get_devices()
    normalized = {str(device).upper().split(".")[0] for device in available_devices}
    if config.device not in normalized:
        raise ConfigurationError(
            f"requested device {config.device} is unavailable; available: "
            f"{', '.join(sorted(normalized)) or '(none)'}"
        )
    if camera_probe is not None and not camera_probe(config.camera_index):
        raise ConfigurationError(
            f"camera index {config.camera_index} could not be opened"
        )
    return config
