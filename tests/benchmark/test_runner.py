"""Benchmark runner tests."""

from pathlib import Path
from typing import Self

import pytest

from financial_volatility.benchmark import (
    BenchmarkRunner,
    Forecast,
    HardwareTarget,
    ModelInput,
    ModelMetadata,
    PredictionContext,
)
from financial_volatility.models import ForecastModel


class DummyForecastModel(ForecastModel):
    """Forecast model with deterministic predictions for runner tests."""

    def __init__(self, predictions: list[float]) -> None:
        """Create a dummy model with fixed prediction values."""
        self.predictions = predictions
        self.train_calls = 0
        self.predict_calls = 0
        self.last_training_data: ModelInput | None = None
        self.last_prediction_context: PredictionContext | None = None
        self.last_horizon: int | None = None

    def train(
        self,
        data: ModelInput,
        validation_data: ModelInput | None = None,
    ) -> None:
        """Record the training call."""
        _ = validation_data
        self.train_calls += 1
        self.last_training_data = data

    def predict(self, context: PredictionContext, horizon: int) -> Forecast:
        """Return fixed forecasts matching the requested horizon."""
        self.predict_calls += 1
        self.last_prediction_context = context
        self.last_horizon = horizon
        return Forecast(values=self.predictions[:horizon], horizon=horizon)

    def save(self, path: str | Path) -> None:
        """Pretend to save model state."""
        _ = path

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Pretend to load model state."""
        _ = path
        return cls(predictions=[])

    def metadata(self) -> ModelMetadata:
        """Return metadata used by benchmark result records."""
        return ModelMetadata(
            name="dummy",
            model_family="test",
            supported_hardware=(HardwareTarget.CPU,),
        )


def test_dummy_model_benchmark_runs_end_to_end() -> None:
    """The runner trains, predicts, times, and evaluates a model."""
    model = DummyForecastModel(predictions=[1.0, 2.0, 5.0])
    training_data = ModelInput(features=[[0.0], [1.0]], target=[0.5, 1.5])
    test_input_data = PredictionContext(features=[[2.0], [3.0], [4.0]])
    test_target_data = [1.0, 2.0, 3.0]

    result = BenchmarkRunner(
        model=model,
        training_data=training_data,
        test_input_data=test_input_data,
        test_target_data=test_target_data,
        hardware_target=HardwareTarget.CPU,
        dataset_name="synthetic",
        target_name="volatility",
    ).run()

    metrics = {metric.name: metric.value for metric in result.metrics}

    assert model.train_calls == 1
    assert model.predict_calls == 1
    assert model.last_training_data == training_data
    assert model.last_prediction_context == test_input_data
    assert model.last_horizon == 3
    assert result.experiment_id == "dummy:synthetic:volatility:h3"
    assert result.model.name == "dummy"
    assert result.dataset_name == "synthetic"
    assert result.target_name == "volatility"
    assert result.horizon == 3
    assert result.hardware == HardwareTarget.CPU
    assert result.forecast is not None
    assert result.forecast.values == [1.0, 2.0, 5.0]
    assert metrics["rmse"] == pytest.approx((4.0 / 3.0) ** 0.5)
    assert metrics["mae"] == pytest.approx(2.0 / 3.0)
    assert metrics["mape"] == pytest.approx(200.0 / 9.0)
    assert metrics["training_time_seconds"] >= 0.0
    assert metrics["inference_time_seconds"] >= 0.0
    assert result.duration_seconds is not None
    assert result.duration_seconds >= 0.0


def test_runner_records_unknown_hardware_label_in_metadata() -> None:
    """Arbitrary device labels are preserved without widening result schema."""
    result = BenchmarkRunner(
        model=DummyForecastModel(predictions=[1.0]),
        training_data=ModelInput(features=[[0.0]], target=[1.0]),
        test_input_data=PredictionContext(features=[[1.0]]),
        test_target_data=[1.0],
        hardware_target="local-cpu",
    ).run()

    assert result.hardware == HardwareTarget.AUTO
    assert result.metadata["hardware_label"] == "local-cpu"


def test_runner_rejects_empty_test_targets() -> None:
    """The runner needs at least one target value to infer forecast horizon."""
    runner = BenchmarkRunner(
        model=DummyForecastModel(predictions=[]),
        training_data=ModelInput(features=[], target=[]),
        test_input_data=PredictionContext(features=[]),
        test_target_data=[],
    )

    with pytest.raises(ValueError, match="at least one"):
        runner.run()
