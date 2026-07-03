"""Model registry tests."""

from pathlib import Path
from typing import ClassVar, Self

import pytest

from financial_volatility.benchmark.types import (
    Forecast,
    HardwareTarget,
    ModelInput,
    ModelMetadata,
    PredictionContext,
)
from financial_volatility.models import ForecastModel, ModelRegistry


class DummyRegistryModel(ForecastModel):
    """Small model adapter used to verify registry behavior."""

    created_parameters: ClassVar[list[dict[str, object]]] = []

    def __init__(self, alpha: float = 1.0, label: str = "dummy") -> None:
        """Record constructor parameters."""
        self.alpha = alpha
        self.label = label
        self.created_parameters.append({"alpha": alpha, "label": label})

    def train(
        self,
        data: ModelInput,
        validation_data: ModelInput | None = None,
    ) -> None:
        """No-op train method."""
        _ = data, validation_data

    def predict(self, context: PredictionContext, horizon: int) -> Forecast:
        """Return deterministic forecasts."""
        _ = context
        return Forecast(values=[self.alpha] * horizon, horizon=horizon)

    def save(self, path: str | Path) -> None:
        """No-op persistence."""
        _ = path

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Return a default dummy model."""
        _ = path
        return cls()

    def metadata(self) -> ModelMetadata:
        """Return stable test metadata."""
        return ModelMetadata(
            name=self.label,
            model_family="test",
            supported_hardware=(HardwareTarget.CPU,),
        )


def test_registry_can_create_registered_model_with_parameters() -> None:
    """Registered model classes can be created with constructor parameters."""
    registry = _fresh_registry()
    registry.register("dummy", DummyRegistryModel)

    model = registry.create("dummy", parameters={"alpha": 0.5, "label": "custom"})

    assert isinstance(model, DummyRegistryModel)
    assert model.alpha == 0.5
    assert model.label == "custom"


def test_registry_normalizes_names() -> None:
    """Names are normalized for lookup."""
    registry = _fresh_registry()
    registry.register("Dummy", DummyRegistryModel)

    model = registry.create(" dummy ")

    assert isinstance(model, DummyRegistryModel)


def test_registry_rejects_unknown_model() -> None:
    """Unknown model names fail with a clear error."""
    registry = _fresh_registry()

    with pytest.raises(ValueError, match="Unknown model"):
        registry.create("missing")


def test_registry_rejects_duplicate_without_replace() -> None:
    """Duplicate registrations require explicit replacement."""
    registry = _fresh_registry()
    registry.register("dummy", DummyRegistryModel)

    with pytest.raises(ValueError, match="already registered"):
        registry.register("dummy", DummyRegistryModel)


def test_registry_lists_names_in_deterministic_order() -> None:
    """Registered names are returned sorted."""
    registry = _fresh_registry()
    registry.register("zeta", DummyRegistryModel)
    registry.register("alpha", DummyRegistryModel)

    assert registry.names() == ("alpha", "zeta")


def test_default_registry_contains_implemented_models() -> None:
    """Built-in adapters are registered by default."""
    assert {"garch", "linear_regression", "xgboost"}.issubset(ModelRegistry.names())
    assert ModelRegistry.create("xgboost").metadata().name == "xgboost"


def _fresh_registry() -> type[ModelRegistry]:
    """Create an isolated registry subclass for tests."""

    class FreshModelRegistry(ModelRegistry):
        _models: ClassVar[dict[str, type[ForecastModel]]] = {}

    return FreshModelRegistry
