"""Validated configuration schemas for benchmark experiments."""

from datetime import date
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ExtensibleConfigModel(BaseModel):
    """Base model that preserves unknown fields for future experiment settings."""

    model_config = ConfigDict(extra="allow")


class DatasetConfig(ExtensibleConfigModel):
    """Dataset selection and date range."""

    ticker: str = Field(min_length=1)
    start_date: date
    end_date: date

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        """Normalize ticker symbols for stable downstream identifiers."""
        return value.strip().upper()

    @model_validator(mode="after")
    def validate_date_range(self) -> "DatasetConfig":
        """Ensure the dataset range is chronological."""
        if self.end_date < self.start_date:
            msg = "end_date must be greater than or equal to start_date"
            raise ValueError(msg)
        return self


class TargetConfig(ExtensibleConfigModel):
    """Forecast target definition."""

    name: str = Field(min_length=1)
    horizon: int = Field(gt=0)


class ModelConfig(ExtensibleConfigModel):
    """Model identifier and model-specific parameter payload."""

    name: str = Field(min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)


class HardwareConfig(ExtensibleConfigModel):
    """Hardware execution preferences."""

    device: str = Field(min_length=1)


class OutputConfig(ExtensibleConfigModel):
    """Result output settings."""

    directory: Path


class BenchmarkConfig(ExtensibleConfigModel):
    """Top-level benchmark configuration."""

    dataset: DatasetConfig
    target: TargetConfig
    model: ModelConfig
    hardware: HardwareConfig
    output: OutputConfig
