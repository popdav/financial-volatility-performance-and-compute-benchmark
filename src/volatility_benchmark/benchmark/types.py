"""Shared benchmark type definitions."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeAlias

ScalarValue: TypeAlias = str | int | float | bool | None
MetadataValue: TypeAlias = (
    ScalarValue | Sequence[ScalarValue] | Mapping[str, ScalarValue]
)


class HardwareTarget(StrEnum):
    """Execution hardware targets that model adapters may support."""

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    TPU = "tpu"


@dataclass(frozen=True, slots=True)
class ModelInput:
    """Model-ready data supplied to a forecasting adapter.

    The payloads are intentionally typed as ``object`` so adapters can accept
    pandas objects, NumPy arrays, PyTorch tensors, or library-specific time
    series structures without making those packages core dependencies.
    """

    features: object
    target: object
    timestamps: Sequence[object] | None = None
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PredictionContext:
    """Data and metadata available when a model generates forecasts."""

    features: object
    timestamps: Sequence[object] | None = None
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Forecast:
    """Forecast values returned by a model adapter."""

    values: object
    horizon: int
    timestamps: Sequence[object] | None = None
    intervals: Mapping[str, object] = field(default_factory=dict)
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """Reproducibility and capability metadata for a model adapter."""

    name: str
    model_family: str
    version: str | None = None
    parameters: Mapping[str, MetadataValue] = field(default_factory=dict)
    supports_probabilistic_forecast: bool = False
    supported_hardware: tuple[HardwareTarget, ...] = (HardwareTarget.CPU,)
