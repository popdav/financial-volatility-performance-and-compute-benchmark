"""Forecast model interface tests."""

from pathlib import Path
from typing import Self

import pytest

from volatility_benchmark.benchmark.types import (
    Forecast,
    HardwareTarget,
    ModelInput,
    ModelMetadata,
    PredictionContext,
)
from volatility_benchmark.models import ForecastModel


class DummyForecastModel(ForecastModel):
    """Minimal concrete model used to verify the abstract interface."""

    def __init__(self) -> None:
        """Create an untrained dummy model."""
        self.is_trained = False

    def train(
        self,
        data: ModelInput,
        validation_data: ModelInput | None = None,
    ) -> None:
        """Record that training was requested."""
        _ = data, validation_data
        self.is_trained = True

    def predict(self, context: PredictionContext, horizon: int) -> Forecast:
        """Return a fixed forecast matching the requested horizon."""
        _ = context
        return Forecast(values=[0.1] * horizon, horizon=horizon)

    def save(self, path: str | Path) -> None:
        """Pretend to persist model state."""
        _ = path

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Pretend to load model state."""
        _ = path
        return cls()

    def metadata(self) -> ModelMetadata:
        """Return stable metadata for assertions."""
        return ModelMetadata(
            name="dummy",
            model_family="test",
            supported_hardware=(HardwareTarget.CPU,),
        )


def test_forecast_model_cannot_be_instantiated_directly() -> None:
    """The abstract base class requires concrete implementations."""
    with pytest.raises(TypeError):
        ForecastModel()


def test_dummy_model_implements_forecast_model() -> None:
    """A concrete adapter can satisfy the benchmark model contract."""
    model = DummyForecastModel()
    train_data = ModelInput(features=[[1.0]], target=[0.2])
    context = PredictionContext(features=[[2.0]])

    model.train(train_data)
    forecast = model.predict(context, horizon=2)
    metadata = model.metadata()

    assert model.is_trained
    assert forecast.values == [0.1, 0.1]
    assert forecast.horizon == 2
    assert metadata.name == "dummy"
    assert metadata.model_family == "test"
    assert metadata.supported_hardware == (HardwareTarget.CPU,)
