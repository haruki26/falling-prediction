"""Windows-friendly, explicit runtime configuration."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias


BedPoints: TypeAlias = tuple[tuple[float, float], ...]


class ConfigurationError(ValueError):
    """Raised when a requested runtime resource is not usable."""


@dataclass(frozen=True)
class AppConfig:
    camera_index: int = 0
    model_path: Path | None = None
    device: str = "CPU"
    bed_left: float | None = None
    bed_top: float | None = None
    bed_right: float | None = None
    bed_bottom: float | None = None
    calibrate: bool = False
    calibration_file: Path = Path("bed_roi.json")


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
    parser.add_argument("--calibrate", action="store_true", help="force interactive bed calibration")
    parser.add_argument("--calibration-file", type=Path, default=Path("bed_roi.json"),
                        help="versioned bed ROI JSON (default: ./bed_roi.json)")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> AppConfig:
    args = build_parser().parse_args(argv)
    return AppConfig(
        args.camera_index,
        args.model_path,
        args.device,
        args.bed_left, args.bed_top, args.bed_right, args.bed_bottom,
        args.calibrate, args.calibration_file,
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
    bed_values = (config.bed_left, config.bed_top, config.bed_right, config.bed_bottom)
    if any(value is not None for value in bed_values) and not all(value is not None for value in bed_values):
        raise ConfigurationError("all four bed overrides must be supplied together")
    if all(value is not None for value in bed_values):
        left, top, right, bottom = bed_values
        assert left is not None and top is not None and right is not None and bottom is not None
        valid_bed = 0 <= left < right <= 1 and 0 <= top < bottom <= 1
    else:
        valid_bed = True
    if not valid_bed:
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


def load_bed_region(path: Path) -> BedPoints | None:
    """Load a versioned normalized four-point ROI.

    Version 1 rectangle files are upgraded in memory to polygon points.
    """
    import json
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        version = data.get("version")
        if version == 1:
            left, top, right, bottom = (float(data[name]) for name in ("left", "top", "right", "bottom"))
            points = ((left, top), (right, top), (right, bottom), (left, bottom))
        elif version == 2:
            points = tuple((float(point[0]), float(point[1])) for point in data["points"])
        else:
            raise ValueError("unsupported version")
        from .risk import BedRegion
        return BedRegion(points=points).points
    except (OSError, ValueError, TypeError, KeyError, AttributeError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"invalid calibration file {path}: {exc}") from exc


def save_bed_region(path: Path, region) -> None:
    """Save a normalized four-point polygon as version 2 JSON.

    A four-number rectangle is accepted for callers using the old API.
    """
    import json
    try:
        if hasattr(region, "points"):
            points = region.points
        elif len(region) == 4 and all(not hasattr(p, "__len__") for p in region):
            left, top, right, bottom = (float(v) for v in region)
            points = ((left, top), (right, top), (right, bottom), (left, bottom))
        else:
            points = region
        from .risk import BedRegion
        points = BedRegion(points=points).points
    except (TypeError, ValueError, IndexError) as exc:
        raise ConfigurationError(f"cannot save invalid normalized bed polygon: {exc}") from exc
    path.write_text(json.dumps({"version": 2, "points": [list(p) for p in points],
                                }, indent=2) + "\n", encoding="utf-8")
