"""Compatibility exports for benchmark types."""

from financial_volatility.benchmark.runner import BenchmarkRunner
from financial_volatility.benchmark.types import (
    Forecast,
    HardwareTarget,
    MetadataValue,
    ModelInput,
    ModelMetadata,
    PredictionContext,
    ScalarValue,
)

__all__ = [
    "BenchmarkRunner",
    "Forecast",
    "HardwareTarget",
    "MetadataValue",
    "ModelInput",
    "ModelMetadata",
    "PredictionContext",
    "ScalarValue",
]
