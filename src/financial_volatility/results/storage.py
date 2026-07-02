"""CSV persistence for benchmark experiment results."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeAlias

from financial_volatility.evaluation.results import ExperimentResult

CSV_COLUMNS = (
    "experiment_id",
    "model_name",
    "model_type",
    "hardware",
    "rmse",
    "mae",
    "mape",
    "training_time_seconds",
    "inference_time_seconds",
    "timestamp",
)

CsvValue: TypeAlias = str | float | None
CsvRow: TypeAlias = dict[str, CsvValue]


def write_experiment_results_csv(
    results: ExperimentResult | Iterable[ExperimentResult],
    path: str | Path,
) -> None:
    """Write one or more experiment results to a CSV file."""
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(_result_to_csv_row(result) for result in _as_iterable(results))


def append_experiment_result_csv(
    result: ExperimentResult,
    path: str | Path,
) -> None:
    """Append one experiment result to a CSV file, writing the header once."""
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    should_write_header = not csv_path.exists() or csv_path.stat().st_size == 0

    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        if should_write_header:
            writer.writeheader()
        writer.writerow(_result_to_csv_row(result))


def _as_iterable(
    results: ExperimentResult | Iterable[ExperimentResult],
) -> Iterable[ExperimentResult]:
    """Normalize a single result or iterable of results."""
    if isinstance(results, ExperimentResult):
        return (results,)
    return results


def _result_to_csv_row(result: ExperimentResult) -> CsvRow:
    """Flatten an experiment result into one stable metrics row."""
    metric_values = {metric.name: metric.value for metric in result.metrics}

    return {
        "experiment_id": result.experiment_id,
        "model_name": result.model.name,
        "model_type": result.model.model_family,
        "hardware": result.hardware.value,
        "rmse": metric_values.get("rmse"),
        "mae": metric_values.get("mae"),
        "mape": metric_values.get("mape"),
        "training_time_seconds": _metric_or_metadata_value(
            metric_values,
            result.metadata,
            "training_time_seconds",
        ),
        "inference_time_seconds": _metric_or_metadata_value(
            metric_values,
            result.metadata,
            "inference_time_seconds",
        ),
        "timestamp": _timestamp_value(result.metadata),
    }


def _metric_or_metadata_value(
    metric_values: Mapping[str, float],
    metadata: Mapping[str, object],
    name: str,
) -> float | None:
    """Read a numeric metric, falling back to numeric metadata."""
    metric_value = metric_values.get(name)
    if metric_value is not None:
        return metric_value

    metadata_value = metadata.get(name)
    if isinstance(metadata_value, int | float):
        return float(metadata_value)

    return None


def _timestamp_value(metadata: Mapping[str, object]) -> str:
    """Use a provided timestamp when present, otherwise record current UTC time."""
    timestamp = metadata.get("timestamp")
    if isinstance(timestamp, str):
        return timestamp

    return datetime.now(UTC).isoformat()
