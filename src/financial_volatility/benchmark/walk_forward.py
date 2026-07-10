"""Walk-forward benchmark orchestration."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from financial_volatility.benchmark.runner import BenchmarkRunner
from financial_volatility.benchmark.types import (
    HardwareTarget,
    MetadataValue,
    ModelInput,
    PredictionContext,
)
from financial_volatility.data import TimeSeriesDataset, WalkForwardSplitter
from financial_volatility.evaluation.results import ExperimentResult, MetricResult
from financial_volatility.models import ForecastModel
from financial_volatility.results.storage import write_experiment_results_csv


@dataclass(frozen=True, slots=True)
class WalkForwardBenchmarkResult:
    """Fold-level and aggregate walk-forward benchmark results."""

    fold_results: tuple[ExperimentResult, ...]
    aggregate_result: ExperimentResult

    @property
    def results(self) -> tuple[ExperimentResult, ...]:
        """Return fold-level results followed by the aggregate result."""
        return self.fold_results + (self.aggregate_result,)


@dataclass(frozen=True, slots=True)
class WalkForwardBenchmark:
    """Run a model factory through all folds from a walk-forward splitter."""

    model_factory: Callable[[], ForecastModel]
    data: pd.DataFrame | TimeSeriesDataset
    target_column: str
    splitter: WalkForwardSplitter
    feature_columns: Sequence[str] | None = None
    hardware_target: HardwareTarget | str = HardwareTarget.AUTO
    dataset_name: str = "default"
    target_name: str = "target"
    forecast_horizon: int | None = None
    results_path: str | Path | None = None

    def run(self) -> WalkForwardBenchmarkResult:
        """Train and evaluate a fresh model on every walk-forward fold."""
        fold_results: list[ExperimentResult] = []

        for fold_index, fold in enumerate(
            self.splitter.split(self.data, name="walk_forward")
        ):
            model = self.model_factory()
            train_frame = fold.train.to_dataframe()
            test_frame = fold.test.to_dataframe()
            result = BenchmarkRunner(
                model=model,
                training_data=ModelInput(
                    features=_feature_frame(
                        train_frame,
                        target_column=self.target_column,
                        feature_columns=self.feature_columns,
                    ),
                    target=train_frame[self.target_column],
                    timestamps=tuple(train_frame.index),
                ),
                test_input_data=PredictionContext(
                    features=_feature_frame(
                        test_frame,
                        target_column=self.target_column,
                        feature_columns=self.feature_columns,
                    ),
                    timestamps=tuple(test_frame.index),
                ),
                test_target_data=test_frame[self.target_column],
                hardware_target=self.hardware_target,
                dataset_name=self.dataset_name,
                target_name=self.target_name,
                forecast_horizon=self.forecast_horizon,
            ).run()
            fold_results.append(_with_fold_metadata(result, fold_index))

        if not fold_results:
            raise ValueError("Walk-forward benchmark produced no folds")

        aggregate_result = _aggregate_results(
            tuple(fold_results),
            dataset_name=self.dataset_name,
            target_name=self.target_name,
        )
        benchmark_result = WalkForwardBenchmarkResult(
            fold_results=tuple(fold_results),
            aggregate_result=aggregate_result,
        )

        if self.results_path is not None:
            write_experiment_results_csv(benchmark_result.results, self.results_path)

        return benchmark_result


def _feature_frame(
    frame: pd.DataFrame,
    *,
    target_column: str,
    feature_columns: Sequence[str] | None,
) -> pd.DataFrame:
    """Select model features from a fold frame."""
    if target_column not in frame.columns:
        raise ValueError(f"target_column is missing from data: {target_column}")

    if feature_columns is None:
        return frame.drop(columns=[target_column])

    missing_columns = [
        column for column in feature_columns if column not in frame.columns
    ]
    if missing_columns:
        msg = f"feature_columns are missing from data: {missing_columns}"
        raise ValueError(msg)

    return frame.loc[:, list(feature_columns)]


def _with_fold_metadata(
    result: ExperimentResult,
    fold_index: int,
) -> ExperimentResult:
    """Add fold identifiers without mutating the benchmark result."""
    metadata: dict[str, MetadataValue] = dict(result.metadata)
    metadata["fold_index"] = fold_index
    return replace(
        result,
        experiment_id=f"{result.experiment_id}:fold{fold_index}",
        metadata=metadata,
    )


def _aggregate_results(
    fold_results: tuple[ExperimentResult, ...],
    *,
    dataset_name: str,
    target_name: str,
) -> ExperimentResult:
    """Aggregate fold metrics by arithmetic mean."""
    first_result = fold_results[0]
    metric_names = sorted(
        {metric.name for result in fold_results for metric in result.metrics}
    )
    aggregate_metrics = tuple(
        _aggregate_metric(metric_name, fold_results) for metric_name in metric_names
    )
    metadata: dict[str, MetadataValue] = {
        "fold_count": len(fold_results),
        "result_level": "aggregate",
    }

    return ExperimentResult(
        experiment_id=(
            f"{first_result.model.name}:{dataset_name}:{target_name}:"
            f"walk_forward:{len(fold_results)}folds"
        ),
        model=first_result.model,
        dataset_name=dataset_name,
        target_name=target_name,
        horizon=first_result.horizon,
        hardware=first_result.hardware,
        metrics=aggregate_metrics,
        forecast=None,
        duration_seconds=_sum_duration_seconds(fold_results),
        metadata=metadata,
    )


def _aggregate_metric(
    metric_name: str,
    fold_results: tuple[ExperimentResult, ...],
) -> MetricResult:
    """Average one metric across all folds that reported it."""
    matching_metrics = [
        metric
        for result in fold_results
        for metric in result.metrics
        if metric.name == metric_name
    ]
    first_metric = matching_metrics[0]
    return MetricResult(
        name=metric_name,
        value=sum(metric.value for metric in matching_metrics) / len(matching_metrics),
        category=first_metric.category,
        higher_is_better=first_metric.higher_is_better,
        metadata={"aggregation": "mean"},
    )


def _sum_duration_seconds(
    fold_results: tuple[ExperimentResult, ...],
) -> float | None:
    """Sum fold durations when every fold reported one."""
    durations = [result.duration_seconds for result in fold_results]
    if any(duration is None for duration in durations):
        return None

    return sum(duration for duration in durations if duration is not None)


__all__ = ["WalkForwardBenchmark", "WalkForwardBenchmarkResult"]
