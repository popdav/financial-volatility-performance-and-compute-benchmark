"""Validated YAML settings for benchmark experiments."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field

HardwareDevice = Literal["auto", "cpu", "cuda", "mps", "tpu"]


class ExtensibleSettings(BaseModel):
    """Base settings model that preserves future configuration fields."""

    model_config = ConfigDict(extra="allow")


class DatasetSettings(ExtensibleSettings):
    """Dataset source and optional date range."""

    provider: str = Field(min_length=1)
    symbols: list[str] = Field(default_factory=list)
    start_date: date | None = None
    end_date: date | None = None


class TargetSettings(ExtensibleSettings):
    """Forecast target definition."""

    name: str = Field(min_length=1)
    horizon: int = Field(gt=0)


class ModelSettings(ExtensibleSettings):
    """Model identifier and model-specific parameter payload."""

    name: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class HardwareSettings(ExtensibleSettings):
    """Hardware execution preference."""

    device: HardwareDevice = "cpu"


class OutputSettings(ExtensibleSettings):
    """Result output settings."""

    directory: Path


class BenchmarkSettings(ExtensibleSettings):
    """Top-level benchmark settings."""

    dataset: DatasetSettings
    target: TargetSettings
    model: ModelSettings
    hardware: HardwareSettings
    output: OutputSettings


DatasetConfig = DatasetSettings
TargetConfig = TargetSettings
ModelConfig = ModelSettings
HardwareConfig = HardwareSettings
OutputConfig = OutputSettings
BenchmarkConfig = BenchmarkSettings


def load_settings(path: str | Path) -> BenchmarkSettings:
    """Load and validate benchmark settings from a YAML file."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as file:
        raw_config = yaml.safe_load(file)

    if not isinstance(raw_config, Mapping):
        msg = f"Configuration file must contain a YAML mapping: {config_path}"
        raise ValueError(msg)

    return BenchmarkSettings.model_validate(raw_config)


def load_config(path: str | Path) -> BenchmarkConfig:
    """Compatibility wrapper for existing config-loading callers."""
    return load_settings(path)


__all__ = [
    "BenchmarkConfig",
    "BenchmarkSettings",
    "DatasetConfig",
    "DatasetSettings",
    "HardwareConfig",
    "HardwareDevice",
    "HardwareSettings",
    "ModelConfig",
    "ModelSettings",
    "OutputConfig",
    "OutputSettings",
    "TargetConfig",
    "TargetSettings",
    "load_config",
    "load_settings",
]
