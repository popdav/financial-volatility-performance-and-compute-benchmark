"""Benchmark orchestration for one forecast model and one test window."""

from __future__ import annotations

from collections.abc import Sized
from dataclasses import dataclass
from time import perf_counter
from typing import cast

import numpy.typing as npt

from financial_volatility.benchmark.profiler import (
    PeakMemoryProfiler,
    estimate_model_size_mb,
)
from financial_volatility.benchmark.types import (
    HardwareTarget,
    MetadataValue,
    ModelInput,
    PredictionContext,
)
from financial_volatility.evaluation.metrics import mae, mape, rmse
from financial_volatility.evaluation.results import ExperimentResult, MetricResult
from financial_volatility.models import ForecastModel


@dataclass(frozen=True, slots=True)
class BenchmarkRunner:
    """Run a single model through train, predict, and accuracy evaluation."""

    model: ForecastModel
    training_data: ModelInput
    test_input_data: PredictionContext
    test_target_data: object
    hardware_target: HardwareTarget | str = HardwareTarget.AUTO
    dataset_name: str = "default"
    target_name: str = "target"

    def run(self) -> ExperimentResult:
        """Train the model, generate forecasts, and return benchmark results."""
        horizon = _infer_horizon(self.test_target_data)
        hardware, hardware_label = _normalize_hardware(self.hardware_target)

        with PeakMemoryProfiler() as memory_profiler:
            training_started = perf_counter()
            self.model.train(self.training_data)
            training_time_seconds = perf_counter() - training_started

            inference_started = perf_counter()
            forecast = self.model.predict(self.test_input_data, horizon=horizon)
            inference_time_seconds = perf_counter() - inference_started
            peak_memory_mb = memory_profiler.peak_memory_mb()

        model_size_mb = estimate_model_size_mb(self.model)
        y_true = cast(npt.ArrayLike, self.test_target_data)
        y_pred = cast(npt.ArrayLike, forecast.values)

        accuracy_metrics = (
            MetricResult(
                name="rmse",
                value=rmse(y_true, y_pred),
                category="accuracy",
                higher_is_better=False,
            ),
            MetricResult(
                name="mae",
                value=mae(y_true, y_pred),
                category="accuracy",
                higher_is_better=False,
            ),
            MetricResult(
                name="mape",
                value=mape(y_true, y_pred),
                category="accuracy",
                higher_is_better=False,
            ),
        )
        compute_metrics = (
            MetricResult(
                name="training_time_seconds",
                value=training_time_seconds,
                category="compute",
                higher_is_better=False,
            ),
            MetricResult(
                name="inference_time_seconds",
                value=inference_time_seconds,
                category="compute",
                higher_is_better=False,
            ),
            MetricResult(
                name="peak_memory_mb",
                value=peak_memory_mb,
                category="compute",
                higher_is_better=False,
            ),
            MetricResult(
                name="model_size_mb",
                value=model_size_mb,
                category="compute",
                higher_is_better=False,
            ),
        )

        model_metadata = self.model.metadata()
        metadata: dict[str, MetadataValue] = {
            "training_time_seconds": training_time_seconds,
            "inference_time_seconds": inference_time_seconds,
            "peak_memory_mb": peak_memory_mb,
            "model_size_mb": model_size_mb,
        }
        if hardware_label is not None:
            metadata["hardware_label"] = hardware_label

        return ExperimentResult(
            experiment_id=(
                f"{model_metadata.name}:{self.dataset_name}:"
                f"{self.target_name}:h{horizon}"
            ),
            model=model_metadata,
            dataset_name=self.dataset_name,
            target_name=self.target_name,
            horizon=horizon,
            hardware=hardware,
            metrics=accuracy_metrics + compute_metrics,
            forecast=forecast,
            duration_seconds=training_time_seconds + inference_time_seconds,
            metadata=metadata,
        )


def _infer_horizon(target_data: object) -> int:
    """Infer forecast horizon from sized target data."""
    if not isinstance(target_data, Sized):
        msg = "test_target_data must be sized so the forecast horizon can be inferred"
        raise TypeError(msg)

    horizon = len(target_data)
    if horizon <= 0:
        raise ValueError("test_target_data must contain at least one target value")

    return horizon


def _normalize_hardware(
    hardware_target: HardwareTarget | str,
) -> tuple[HardwareTarget, str | None]:
    """Return an ExperimentResult hardware enum plus optional raw device label."""
    if isinstance(hardware_target, HardwareTarget):
        return hardware_target, None

    try:
        return HardwareTarget(hardware_target), None
    except ValueError:
        return HardwareTarget.AUTO, hardware_target


__all__ = ["BenchmarkRunner"]
