"""Model registry for configuration-driven model construction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeVar

from financial_volatility.benchmark.types import ScalarValue
from financial_volatility.models.base import ForecastModel
from financial_volatility.models.garch import GARCHModel
from financial_volatility.models.linear import LinearRegressionModel
from financial_volatility.models.lstm import LSTMModel
from financial_volatility.models.transformer import TransformerModel
from financial_volatility.models.xgboost import XGBoostModel

ModelT = TypeVar("ModelT", bound=ForecastModel)


class ModelRegistry:
    """Register and instantiate forecasting model adapters by name."""

    _models: dict[str, type[ForecastModel]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        model_cls: type[ModelT],
        *,
        replace: bool = False,
    ) -> None:
        """Register a model adapter class under a stable name."""
        normalized_name = _normalize_name(name)
        if not issubclass(model_cls, ForecastModel):
            msg = "model_cls must implement ForecastModel"
            raise TypeError(msg)

        if normalized_name in cls._models and not replace:
            msg = f"Model is already registered: {normalized_name}"
            raise ValueError(msg)

        cls._models[normalized_name] = model_cls

    @classmethod
    def create(
        cls,
        name: str,
        parameters: Mapping[str, ScalarValue] | None = None,
    ) -> ForecastModel:
        """Instantiate a registered model by name."""
        normalized_name = _normalize_name(name)
        try:
            model_cls = cls._models[normalized_name]
        except KeyError as error:
            available = ", ".join(cls.names()) or "none"
            msg = f"Unknown model: {normalized_name}. Available models: {available}"
            raise ValueError(msg) from error

        return model_cls(**dict(parameters or {}))

    @classmethod
    def names(cls) -> tuple[str, ...]:
        """Return registered model names in deterministic order."""
        return tuple(sorted(cls._models))

    @classmethod
    def clear(cls) -> None:
        """Remove all registrations, primarily for tests."""
        cls._models.clear()


def register_default_models() -> None:
    """Register built-in model adapters."""
    ModelRegistry.register("garch", GARCHModel, replace=True)
    ModelRegistry.register("linear_regression", LinearRegressionModel, replace=True)
    ModelRegistry.register("lstm", LSTMModel, replace=True)
    ModelRegistry.register("transformer", TransformerModel, replace=True)
    ModelRegistry.register("xgboost", XGBoostModel, replace=True)


def _normalize_name(name: str) -> str:
    """Normalize a model registry key."""
    normalized_name = name.strip().lower()
    if not normalized_name:
        raise ValueError("Model name must be non-empty")

    return normalized_name


register_default_models()

__all__ = ["ModelRegistry", "register_default_models"]
