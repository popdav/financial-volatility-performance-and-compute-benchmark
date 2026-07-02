"""Structured result records produced by future benchmark runs."""

from collections.abc import Mapping
from dataclasses import dataclass, field

from volatility_benchmark.benchmark.types import (
    Forecast,
    HardwareTarget,
    MetadataValue,
    ModelMetadata,
)


@dataclass(frozen=True, slots=True)
class MetricResult:
    """Single metric value associated with one experiment output."""

    name: str
    value: float
    category: str
    higher_is_better: bool | None = None
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Result envelope for one model, dataset, target, and forecast horizon."""

    experiment_id: str
    model: ModelMetadata
    dataset_name: str
    target_name: str
    horizon: int
    hardware: HardwareTarget
    metrics: tuple[MetricResult, ...] = ()
    forecast: Forecast | None = None
    duration_seconds: float | None = None
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)
