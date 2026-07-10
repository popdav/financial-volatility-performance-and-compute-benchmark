"""Minimal experiment pipeline tests."""

from pathlib import Path
from typing import Self

import pandas as pd

from financial_volatility.benchmark import (
    Forecast,
    HardwareTarget,
    ModelInput,
    ModelMetadata,
    PredictionContext,
)
from financial_volatility.features.engineering import build_supervised_dataset
from financial_volatility.models import ForecastModel
from financial_volatility.pipelines.experiment import ExperimentPipeline


class DummyForecastModel(ForecastModel):
    """Forecast model used to verify orchestration without real model logic."""

    def __init__(self) -> None:
        """Create a dummy model."""
        self.train_calls = 0
        self.predict_calls = 0
        self.training_timestamps: tuple[object, ...] = ()
        self.prediction_timestamps: tuple[object, ...] = ()
        self.training_target: pd.Series | None = None

    def train(
        self,
        data: ModelInput,
        validation_data: ModelInput | None = None,
    ) -> None:
        """Record training data passed by the pipeline."""
        _ = validation_data
        self.train_calls += 1
        self.training_timestamps = tuple(data.timestamps or ())
        self.training_target = (
            data.target if isinstance(data.target, pd.Series) else None
        )

    def predict(self, context: PredictionContext, horizon: int) -> Forecast:
        """Return a deterministic forecast with the requested horizon."""
        self.predict_calls += 1
        self.prediction_timestamps = tuple(context.timestamps or ())
        return Forecast(values=[0.01] * horizon, horizon=horizon)

    def save(self, path: str | Path) -> None:
        """Pretend to persist model state."""
        _ = path

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Pretend to load model state."""
        _ = path
        return cls()

    def metadata(self) -> ModelMetadata:
        """Return stable metadata for result records."""
        return ModelMetadata(
            name="dummy",
            model_family="test",
            supported_hardware=(HardwareTarget.CPU,),
        )


def test_experiment_pipeline_runs_end_to_end_with_dummy_model(
    tmp_path: Path,
) -> None:
    """The pipeline loads, features, splits, benchmarks, and writes results."""
    csv_path = tmp_path / "prices.csv"
    results_path = tmp_path / "results" / "metrics.csv"
    _write_synthetic_csv(csv_path)
    model = DummyForecastModel()

    result = ExperimentPipeline(
        csv_path=csv_path,
        model=model,
        results_path=results_path,
        test_size=0.25,
        dataset_name="synthetic",
    ).run()

    metrics = {metric.name: metric.value for metric in result.metrics}
    result_rows = pd.read_csv(results_path)

    assert model.train_calls == 1
    assert model.predict_calls == 1
    assert results_path.exists()
    assert len(result_rows) == 1
    assert result_rows["experiment_id"].iloc[0] == result.experiment_id
    assert {"rmse", "mae", "mape"}.issubset(metrics)
    assert result.target_name == "realized_volatility_target_5d"
    assert result.horizon == 5
    assert result.forecast is not None
    assert len(result.forecast.values) == result.metadata["prediction_count"]
    assert model.training_target is not None
    _features, expected_target = build_supervised_dataset(
        pd.read_csv(csv_path, parse_dates=["date"]).set_index("date").sort_index(),
        horizon=5,
    )
    assert model.training_target.iloc[0] == expected_target.iloc[0]


def test_experiment_pipeline_preserves_chronological_train_test_split(
    tmp_path: Path,
) -> None:
    """The pipeline feeds chronological train and test windows to the model."""
    csv_path = tmp_path / "prices.csv"
    results_path = tmp_path / "results.csv"
    _write_synthetic_csv(csv_path)
    model = DummyForecastModel()

    ExperimentPipeline(
        csv_path=csv_path,
        model=model,
        results_path=results_path,
        test_size=0.25,
    ).run()

    train_index = pd.DatetimeIndex(model.training_timestamps)
    test_index = pd.DatetimeIndex(model.prediction_timestamps)

    assert train_index.is_monotonic_increasing
    assert test_index.is_monotonic_increasing
    assert train_index[-1] < test_index[0]


def _write_synthetic_csv(path: Path) -> None:
    """Write enough OHLCV rows for feature warm-up and splitting."""
    dates = pd.date_range("2026-01-01", periods=32, freq="D")
    close = [100.0 + index for index in range(len(dates))]
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": [price + 1.0 for price in close],
            "low": [price - 1.0 for price in close],
            "close": close,
            "volume": [1000 + index for index in range(len(dates))],
        },
    )
    frame = frame.sample(frac=1.0, random_state=1)
    frame.to_csv(path, index=False)
