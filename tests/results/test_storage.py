"""CSV result storage tests."""

import csv
from pathlib import Path

from financial_volatility.benchmark import HardwareTarget, ModelMetadata
from financial_volatility.evaluation.results import ExperimentResult, MetricResult
from financial_volatility.results.storage import (
    CSV_COLUMNS,
    append_experiment_result_csv,
    write_experiment_results_csv,
)


def test_csv_export_works_for_one_result(tmp_path: Path) -> None:
    """A single experiment result is flattened into one CSV row."""
    path = tmp_path / "nested" / "results.csv"

    write_experiment_results_csv(_result("exp-1"), path)

    rows = _read_rows(path)
    assert path.exists()
    assert len(rows) == 1
    assert rows[0] == {
        "experiment_id": "exp-1",
        "model_name": "dummy",
        "model_type": "test",
        "hardware": "cpu",
        "rmse": "1.5",
        "mae": "1.0",
        "mape": "10.0",
        "training_time_seconds": "0.25",
        "inference_time_seconds": "0.05",
        "timestamp": "2026-07-02T12:00:00+00:00",
    }


def test_csv_export_works_for_multiple_results(tmp_path: Path) -> None:
    """Multiple experiment results are written under one header."""
    path = tmp_path / "results.csv"

    write_experiment_results_csv([_result("exp-1"), _result("exp-2")], path)

    rows = _read_rows(path)
    assert [row["experiment_id"] for row in rows] == ["exp-1", "exp-2"]
    assert _header(path) == list(CSV_COLUMNS)


def test_csv_append_works_without_duplicating_headers(tmp_path: Path) -> None:
    """Appending results preserves exactly one CSV header."""
    path = tmp_path / "results" / "benchmark.csv"

    append_experiment_result_csv(_result("exp-1"), path)
    append_experiment_result_csv(_result("exp-2"), path)

    rows = _read_rows(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    header_line = ",".join(CSV_COLUMNS)

    assert [row["experiment_id"] for row in rows] == ["exp-1", "exp-2"]
    assert lines.count(header_line) == 1


def test_csv_export_accepts_single_result_argument(tmp_path: Path) -> None:
    """The write helper accepts one result as well as iterables."""
    path = tmp_path / "results.csv"

    write_experiment_results_csv(_result("exp-1"), path)

    assert _read_rows(path)[0]["experiment_id"] == "exp-1"


def test_csv_export_falls_back_to_timing_metadata(tmp_path: Path) -> None:
    """Timing columns can come from result metadata when metric rows are absent."""
    path = tmp_path / "results.csv"
    result = ExperimentResult(
        experiment_id="exp-1",
        model=ModelMetadata(name="dummy", model_family="test"),
        dataset_name="synthetic",
        target_name="volatility",
        horizon=1,
        hardware=HardwareTarget.CPU,
        metrics=(
            MetricResult("rmse", 1.5, "accuracy", False),
            MetricResult("mae", 1.0, "accuracy", False),
            MetricResult("mape", 10.0, "accuracy", False),
        ),
        metadata={
            "training_time_seconds": 0.25,
            "inference_time_seconds": 0.05,
            "timestamp": "2026-07-02T12:00:00+00:00",
        },
    )

    write_experiment_results_csv(result, path)

    row = _read_rows(path)[0]
    assert row["training_time_seconds"] == "0.25"
    assert row["inference_time_seconds"] == "0.05"


def _result(experiment_id: str) -> ExperimentResult:
    """Create a representative experiment result."""
    return ExperimentResult(
        experiment_id=experiment_id,
        model=ModelMetadata(name="dummy", model_family="test"),
        dataset_name="synthetic",
        target_name="volatility",
        horizon=3,
        hardware=HardwareTarget.CPU,
        metrics=(
            MetricResult("rmse", 1.5, "accuracy", False),
            MetricResult("mae", 1.0, "accuracy", False),
            MetricResult("mape", 10.0, "accuracy", False),
            MetricResult("training_time_seconds", 0.25, "compute", False),
            MetricResult("inference_time_seconds", 0.05, "compute", False),
        ),
        metadata={"timestamp": "2026-07-02T12:00:00+00:00"},
    )


def _read_rows(path: Path) -> list[dict[str, str]]:
    """Read CSV rows as dictionaries."""
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _header(path: Path) -> list[str]:
    """Read the CSV header row."""
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        return next(reader)
