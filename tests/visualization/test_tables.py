"""Thesis table generation tests."""

import pandas as pd
import pytest

from financial_volatility.visualization import (
    ThesisTablePaths,
    generate_thesis_tables,
    summarize_results,
)


def test_summarize_results_is_deterministic() -> None:
    """Summary output is sorted and aggregates repeated model rows."""
    summary = summarize_results(_results_frame())

    assert summary["model_name"].tolist() == ["linear_regression", "xgboost"]
    assert summary.loc[0, "rmse"] == pytest.approx(0.25)
    assert summary.loc[1, "training_time_seconds"] == pytest.approx(0.03)


def test_generate_thesis_tables_writes_markdown_csv_and_latex(tmp_path) -> None:
    """Table generation writes all requested output formats."""
    results_csv = tmp_path / "results.csv"
    _results_frame().to_csv(results_csv, index=False)

    paths = generate_thesis_tables(
        results_csv,
        tmp_path / "tables",
        include_latex=True,
    )

    assert isinstance(paths, ThesisTablePaths)
    assert paths.markdown.exists()
    assert paths.csv.exists()
    assert paths.latex is not None
    assert paths.latex.exists()
    assert "linear_regression" in paths.markdown.read_text(encoding="utf-8")


def test_generate_thesis_tables_rejects_empty_results(tmp_path) -> None:
    """Empty result CSV files fail clearly."""
    results_csv = tmp_path / "results.csv"
    pd.DataFrame(columns=_results_frame().columns).to_csv(results_csv, index=False)

    with pytest.raises(ValueError, match="at least one row"):
        generate_thesis_tables(results_csv, tmp_path / "tables")


def _results_frame() -> pd.DataFrame:
    """Create representative benchmark result rows."""
    return pd.DataFrame(
        {
            "experiment_id": [
                "linear:synthetic:h1",
                "linear:synthetic:h2",
                "xgboost:synthetic:h1",
            ],
            "model_name": ["xgboost", "linear_regression", "linear_regression"],
            "model_type": [
                "machine_learning",
                "statistical_baseline",
                "statistical_baseline",
            ],
            "hardware": ["cpu", "cpu", "cpu"],
            "rmse": [0.15, 0.2, 0.3],
            "mae": [0.08, 0.1, 0.12],
            "mape": [4.0, 5.0, 6.0],
            "training_time_seconds": [0.03, 0.01, 0.02],
            "inference_time_seconds": [0.002, 0.001, 0.001],
            "peak_memory_mb": [2.0, 1.0, 1.2],
            "model_size_mb": [0.03, 0.01, 0.01],
            "timestamp": ["2026-07-02T12:00:00+00:00"] * 3,
        }
    )
