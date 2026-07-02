"""Linear regression model adapter tests."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from financial_volatility.benchmark import BenchmarkRunner
from financial_volatility.benchmark.types import (
    HardwareTarget,
    ModelInput,
    PredictionContext,
)
from financial_volatility.models import ForecastModel, LinearRegressionModel


def test_linear_regression_model_can_be_instantiated() -> None:
    """The adapter exposes stable metadata."""
    model = LinearRegressionModel()

    metadata = model.metadata()

    assert metadata.name == "linear_regression"
    assert metadata.model_family == "statistical_baseline"
    assert metadata.parameters == {}


def test_linear_regression_model_implements_forecast_model() -> None:
    """LinearRegressionModel satisfies the shared forecasting interface."""
    assert isinstance(LinearRegressionModel(), ForecastModel)


def test_linear_regression_model_can_train_on_synthetic_features() -> None:
    """Synthetic tabular feature data can fit the adapter."""
    model = LinearRegressionModel()

    model.train(_training_input())

    forecast = model.predict(
        PredictionContext(features=_feature_frame().iloc[:3]),
        horizon=3,
    )
    assert len(forecast.values) == 3


def test_linear_regression_model_can_produce_predictions() -> None:
    """Predictions follow the requested horizon and are finite."""
    model = LinearRegressionModel()
    model.train(_training_input())

    forecast = model.predict(
        PredictionContext(features=_feature_frame().iloc[10:15]),
        horizon=5,
    )

    values = np.asarray(forecast.values, dtype=np.float64)
    assert forecast.horizon == 5
    assert len(values) == 5
    assert np.all(np.isfinite(values))


def test_linear_regression_model_rejects_prediction_before_training() -> None:
    """Prediction requires a fitted estimator."""
    with pytest.raises(ValueError, match="trained"):
        LinearRegressionModel().predict(
            PredictionContext(features=_feature_frame().iloc[:1]),
            horizon=1,
        )


def test_linear_regression_model_save_and_load_works(tmp_path: Path) -> None:
    """A trained adapter can be persisted and restored."""
    path = tmp_path / "linear.joblib"
    model = LinearRegressionModel()
    model.train(_training_input())

    expected = model.predict(
        PredictionContext(features=_feature_frame().iloc[:4]),
        horizon=4,
    )
    model.save(path)
    loaded = LinearRegressionModel.load(path)
    actual = loaded.predict(
        PredictionContext(features=_feature_frame().iloc[:4]),
        horizon=4,
    )

    assert actual.values == pytest.approx(expected.values)


def test_benchmark_runner_can_evaluate_linear_regression_model() -> None:
    """BenchmarkRunner can train, forecast, and score the adapter."""
    model = LinearRegressionModel()
    features = _feature_frame()
    target = _target_series(features)
    train_features = features.iloc[:25]
    test_features = features.iloc[25:30]
    test_target = target.iloc[25:30]

    result = BenchmarkRunner(
        model=model,
        training_data=ModelInput(
            features=train_features,
            target=target.iloc[:25],
            timestamps=tuple(train_features.index),
        ),
        test_input_data=PredictionContext(
            features=test_features,
            timestamps=tuple(test_features.index),
        ),
        test_target_data=test_target,
        hardware_target=HardwareTarget.CPU,
        dataset_name="synthetic",
        target_name="realized_volatility",
    ).run()

    metrics = {metric.name: metric.value for metric in result.metrics}

    assert result.model.name == "linear_regression"
    assert result.model.model_family == "statistical_baseline"
    assert result.horizon == len(test_target)
    assert result.forecast is not None
    assert len(result.forecast.values) == len(test_target)
    assert {"rmse", "mae", "mape"}.issubset(metrics)


def _training_input() -> ModelInput:
    """Create synthetic model input with linear target structure."""
    features = _feature_frame()
    target = _target_series(features)
    return ModelInput(
        features=features,
        target=target,
        timestamps=tuple(features.index),
    )


def _feature_frame() -> pd.DataFrame:
    """Create deterministic synthetic tabular features."""
    index = pd.date_range("2026-01-01", periods=40, freq="D")
    x1 = np.linspace(0.0, 1.0, num=40)
    x2 = np.linspace(1.0, 2.0, num=40)
    return pd.DataFrame(
        {
            "return_lag_1": x1,
            "volatility_lag_1": x2,
            "ma_5": x1 + x2,
        },
        index=index,
    )


def _target_series(features: pd.DataFrame) -> pd.Series:
    """Create a synthetic realized volatility target."""
    return pd.Series(
        0.01 + 0.02 * features["return_lag_1"] + 0.03 * features["volatility_lag_1"],
        index=features.index,
        name="realized_volatility_5d",
    )
