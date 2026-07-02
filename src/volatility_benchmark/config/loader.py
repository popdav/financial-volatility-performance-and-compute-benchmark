"""YAML configuration loading."""

from collections.abc import Mapping
from pathlib import Path

import yaml  # type: ignore[import-untyped]

from volatility_benchmark.config.schema import BenchmarkConfig


def load_config(path: str | Path) -> BenchmarkConfig:
    """Load and validate a benchmark configuration from a YAML file."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as file:
        raw_config = yaml.safe_load(file)

    if not isinstance(raw_config, Mapping):
        msg = f"Configuration file must contain a YAML mapping: {config_path}"
        raise ValueError(msg)

    return BenchmarkConfig.model_validate(raw_config)
