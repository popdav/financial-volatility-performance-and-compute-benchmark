"""Walk-forward benchmark tests."""

from pathlib import Path

import numpy as np
import pandas as pd

from financial_volatility.benchmark import HardwareTarget, WalkForwardBenchmark
from financial_volatility.data import WalkForwardSplitter
from financial_volatility.models import GARCHModel


def test_garch_walk_forward_benchmark_runs_and_aggregates() -> None:
    """GARCH is retrained per fold and aggregate metrics are returned."""
    data = _garch_frame()

    result = WalkForwardBenchmark(
        model_factory=GARCHModel,
        data=data,
        target_column="realized_volatility",
        splitter=WalkForwardSplitter(
            initial_train_size=80,
            test_window_size=3,
            step_size=20,
        ),
        dataset_name="synthetic",
        target_name="realized_volatility",
        hardware_target=HardwareTarget.CPU,
    ).run()

    aggregate_metrics = {
        metric.name: metric.value for metric in result.aggregate_result.metrics
    }

    assert len(result.fold_results) == 5
    assert result.aggregate_result.metadata["fold_count"] == 5
    assert result.aggregate_result.model.name == "garch"
    assert {"rmse", "mae", "mape"}.issubset(aggregate_metrics)
    assert all(fold.forecast is not None for fold in result.fold_results)
    assert all(
        fold.metadata["fold_index"] == index
        for index, fold in enumerate(result.fold_results)
    )


def test_garch_walk_forward_benchmark_saves_fold_and_aggregate_results(
    tmp_path: Path,
) -> None:
    """Fold-level and aggregate results can be persisted to CSV."""
    path = tmp_path / "walk_forward.csv"

    result = WalkForwardBenchmark(
        model_factory=GARCHModel,
        data=_garch_frame(rows=100),
        target_column="realized_volatility",
        splitter=WalkForwardSplitter(
            initial_train_size=70,
            test_window_size=2,
            step_size=20,
        ),
        dataset_name="synthetic",
        target_name="realized_volatility",
        results_path=path,
    ).run()

    saved = pd.read_csv(path)

    assert path.exists()
    assert len(saved) == len(result.fold_results) + 1
    assert saved["model_name"].tolist()[-1] == "garch"


def _garch_frame(rows: int = 170) -> pd.DataFrame:
    """Create deterministic returns and volatility targets for GARCH tests."""
    rng = np.random.default_rng(42)
    index = pd.date_range("2026-01-01", periods=rows, freq="D")
    volatility = np.linspace(0.006, 0.018, num=rows)
    returns = rng.normal(loc=0.0, scale=1.0, size=rows) * volatility
    return pd.DataFrame(
        {
            "log_return_1d": returns,
            "realized_volatility": np.abs(returns),
        },
        index=index,
    )
