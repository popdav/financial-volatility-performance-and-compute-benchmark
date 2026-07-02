"""GARCH model adapter tests."""

import numpy as np
import pandas as pd
import pytest

from financial_volatility.benchmark import BenchmarkRunner
from financial_volatility.benchmark.types import (
    Forecast,
    HardwareTarget,
    ModelInput,
    PredictionContext,
)
from financial_volatility.models import ForecastModel, GARCHModel


def test_garch_model_can_be_instantiated() -> None:
    """The default GARCH model is GARCH(1, 1)."""
    model = GARCHModel()

    metadata = model.metadata()

    assert metadata.name == "garch"
    assert metadata.model_family == "statistical"
    assert metadata.parameters == {"p": 1, "q": 1}


def test_garch_model_implements_forecast_model() -> None:
    """GARCHModel satisfies the shared forecasting interface."""
    assert isinstance(GARCHModel(), ForecastModel)


def test_garch_model_can_train_on_synthetic_returns() -> None:
    """Synthetic log returns can be used to fit the model."""
    model = GARCHModel()

    model.train(_training_input())

    forecast = model.predict(PredictionContext(features=pd.DataFrame()), horizon=1)
    assert isinstance(forecast, Forecast)
    assert len(forecast.values) == 1


def test_garch_model_produces_non_empty_volatility_forecasts() -> None:
    """Forecasts are positive volatility values with the requested horizon."""
    model = GARCHModel()
    model.train(_training_input())

    forecast = model.predict(PredictionContext(features=pd.DataFrame()), horizon=5)

    values = np.asarray(forecast.values, dtype=np.float64)
    assert forecast.horizon == 5
    assert len(values) == 5
    assert np.all(np.isfinite(values))
    assert np.all(values >= 0.0)


def test_garch_model_can_train_from_target_when_return_column_is_absent() -> None:
    """The adapter falls back to ModelInput.target for direct return inputs."""
    model = GARCHModel()
    returns = _synthetic_returns()

    model.train(ModelInput(features=pd.DataFrame({"x": returns}), target=returns))

    forecast = model.predict(PredictionContext(features=pd.DataFrame()), horizon=2)
    assert len(forecast.values) == 2


def test_garch_model_requires_training_before_prediction() -> None:
    """Prediction before fitting is rejected with a clear error."""
    with pytest.raises(ValueError, match="trained"):
        GARCHModel().predict(PredictionContext(features=pd.DataFrame()), horizon=1)


def test_benchmark_runner_can_evaluate_garch_model() -> None:
    """BenchmarkRunner can train, forecast, and score the GARCH adapter."""
    model = GARCHModel()
    target = [0.008, 0.009, 0.01]

    result = BenchmarkRunner(
        model=model,
        training_data=_training_input(),
        test_input_data=PredictionContext(
            features=pd.DataFrame(index=pd.RangeIndex(len(target))),
        ),
        test_target_data=target,
        hardware_target=HardwareTarget.CPU,
        dataset_name="synthetic",
        target_name="volatility",
    ).run()

    metrics = {metric.name: metric.value for metric in result.metrics}

    assert result.model.name == "garch"
    assert result.model.model_family == "statistical"
    assert result.horizon == len(target)
    assert result.forecast is not None
    assert len(result.forecast.values) == len(target)
    assert {"rmse", "mae", "mape"}.issubset(metrics)


def _training_input() -> ModelInput:
    """Create model input containing synthetic log returns."""
    returns = _synthetic_returns()
    features = pd.DataFrame({"log_return_1d": returns})
    return ModelInput(features=features, target=np.abs(returns))


def _synthetic_returns() -> np.ndarray:
    """Generate deterministic heteroskedastic returns."""
    rng = np.random.default_rng(42)
    shocks = rng.normal(loc=0.0, scale=1.0, size=180)
    volatility = np.linspace(0.006, 0.018, num=180)
    return shocks * volatility
