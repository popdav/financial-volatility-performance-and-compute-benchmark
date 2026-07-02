"""Framework-agnostic forecasting model contracts."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

from volatility_benchmark.benchmark.types import (
    Forecast,
    ModelInput,
    ModelMetadata,
    PredictionContext,
)


class ForecastModel(ABC):
    """Abstract contract implemented by all volatility forecasting adapters.

    The interface represents the benchmark-facing behavior only. Concrete
    adapters may wrap statistical libraries, scikit-learn estimators, PyTorch
    modules, or future hardware-specific runtimes behind these methods.
    """

    @abstractmethod
    def train(
        self,
        data: ModelInput,
        validation_data: ModelInput | None = None,
    ) -> None:
        """Fit or estimate model state from benchmark-provided training data."""

    @abstractmethod
    def predict(self, context: PredictionContext, horizon: int) -> Forecast:
        """Generate volatility forecasts for the requested forecast horizon."""

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist enough model state to reconstruct the adapter later."""

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> Self:
        """Load a previously persisted model adapter."""

    @abstractmethod
    def metadata(self) -> ModelMetadata:
        """Return reproducibility metadata for benchmark result records."""
