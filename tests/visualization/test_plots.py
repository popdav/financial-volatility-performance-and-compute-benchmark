"""Result plot generation tests."""

import pandas as pd
import pytest

from financial_volatility.visualization import PLOT_FILENAMES, generate_result_plots


def test_generate_result_plots_creates_png_files(tmp_path) -> None:
    """Plot generation writes every expected PNG file."""
    results_csv = tmp_path / "results.csv"
    output_dir = tmp_path / "plots"
    _results_frame().to_csv(results_csv, index=False)

    paths = generate_result_plots(results_csv, output_dir)

    assert len(paths) == len(PLOT_FILENAMES)
    assert {path.name for path in paths} == set(PLOT_FILENAMES.values())
    assert all(path.exists() for path in paths)
    assert all(path.stat().st_size > 0 for path in paths)


def test_generate_result_plots_rejects_empty_results(tmp_path) -> None:
    """Empty result CSV files fail clearly."""
    results_csv = tmp_path / "results.csv"
    pd.DataFrame(columns=_results_frame().columns).to_csv(results_csv, index=False)

    with pytest.raises(ValueError, match="at least one row"):
        generate_result_plots(results_csv, tmp_path / "plots")


def _results_frame() -> pd.DataFrame:
    """Create representative benchmark result rows."""
    return pd.DataFrame(
        {
            "experiment_id": ["linear:synthetic:h1", "xgboost:synthetic:h1"],
            "model_name": ["linear_regression", "xgboost"],
            "model_type": ["statistical_baseline", "machine_learning"],
            "hardware": ["cpu", "cpu"],
            "rmse": [0.2, 0.15],
            "mae": [0.1, 0.08],
            "mape": [5.0, 4.0],
            "training_time_seconds": [0.01, 0.03],
            "inference_time_seconds": [0.001, 0.002],
            "peak_memory_mb": [1.0, 2.0],
            "model_size_mb": [0.01, 0.03],
            "timestamp": ["2026-07-02T12:00:00+00:00"] * 2,
        }
    )
