"""Transformer model adapter tests."""

from pathlib import Path

import numpy as np
import pytest

from financial_volatility.benchmark import BenchmarkRunner
from financial_volatility.benchmark.types import (
    HardwareTarget,
    ModelInput,
    PredictionContext,
)
from financial_volatility.models import ForecastModel, TransformerModel


def test_transformer_model_can_be_instantiated() -> None:
    """The adapter exposes stable metadata."""
    model = _model()

    metadata = model.metadata()

    assert metadata.name == "transformer"
    assert metadata.model_family == "deep_learning"
    assert metadata.parameters["sequence_length"] == 4
    assert metadata.parameters["model_dimension"] == 4


def test_transformer_model_implements_forecast_model() -> None:
    """TransformerModel satisfies the shared forecasting interface."""
    assert isinstance(_model(), ForecastModel)


def test_transformer_model_trains_and_predicts_sequence_data() -> None:
    """Synthetic sequence data can fit the adapter and produce forecasts."""
    model = _model()
    features, target = _sequence_data()

    model.train(ModelInput(features=features[:24], target=target[:24]))
    forecast = model.predict(PredictionContext(features=features[24:29]), horizon=5)

    values = np.asarray(forecast.values, dtype=np.float64)
    assert forecast.horizon == 5
    assert len(values) == 5
    assert np.all(np.isfinite(values))


def test_transformer_model_requires_training_before_prediction() -> None:
    """Prediction requires trained weights."""
    features, _target = _sequence_data()

    with pytest.raises(ValueError, match="trained"):
        _model().predict(PredictionContext(features=features[:1]), horizon=1)


def test_transformer_model_save_and_load_works(tmp_path: Path) -> None:
    """A trained adapter can be persisted and restored."""
    path = tmp_path / "transformer.pt"
    model = _model()
    features, target = _sequence_data()
    model.train(ModelInput(features=features[:24], target=target[:24]))

    expected = model.predict(PredictionContext(features=features[24:28]), horizon=4)
    model.save(path)
    loaded = TransformerModel.load(path)
    actual = loaded.predict(PredictionContext(features=features[24:28]), horizon=4)

    assert actual.values == pytest.approx(expected.values)


def test_benchmark_runner_can_evaluate_transformer_model() -> None:
    """BenchmarkRunner can train, forecast, and score the adapter."""
    features, target = _sequence_data()
    train_features = features[:24]
    test_features = features[24:29]
    test_target = target[24:29]

    result = BenchmarkRunner(
        model=_model(),
        training_data=ModelInput(features=train_features, target=target[:24]),
        test_input_data=PredictionContext(features=test_features),
        test_target_data=test_target,
        hardware_target=HardwareTarget.CPU,
        dataset_name="synthetic",
        target_name="volatility",
    ).run()

    metrics = {metric.name: metric.value for metric in result.metrics}

    assert result.model.name == "transformer"
    assert result.model.model_family == "deep_learning"
    assert result.horizon == len(test_target)
    assert result.forecast is not None
    assert len(result.forecast.values) == len(test_target)
    assert {"rmse", "mae", "mape"}.issubset(metrics)


def _model() -> TransformerModel:
    """Create a small deterministic Transformer adapter for tests."""
    return TransformerModel(
        sequence_length=4,
        model_dimension=4,
        num_heads=2,
        epochs=2,
        learning_rate=0.01,
        device="cpu",
    )


def _sequence_data() -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic synthetic sequence data."""
    rng = np.random.default_rng(42)
    features = rng.normal(size=(32, 4, 3)).astype(np.float32)
    target = (
        0.1 * features[:, -1, 0]
        + 0.2 * features[:, -1, 1]
        - 0.05 * features[:, -1, 2]
        + 0.5
    ).astype(np.float32)
    return features, target
