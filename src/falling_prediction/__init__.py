"""Core building blocks for the falling-prediction MVP."""

from .config import AppConfig, ConfigurationError, parse_args, validate_config
from .risk import RiskEvaluator, RiskLevel, RiskResult

__all__ = [
    "AppConfig",
    "ConfigurationError",
    "RiskEvaluator",
    "RiskLevel",
    "RiskResult",
    "main",
    "parse_args",
    "validate_config",
]


def main() -> None:
    """Run the Windows CPU webcam MVP."""
    config = parse_args()
    validate_config(config)
    from .app import run

    run(config)
