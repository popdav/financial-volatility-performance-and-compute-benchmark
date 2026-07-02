"""Linear regression forecasting model adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Self, cast

import joblib  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.linear_model import LinearRegression  # type: ignore[import-untyped]

from financial_volatility.benchmark.types import (
    Forecast,
    HardwareTarget,
    ModelInput,
    ModelMetadata,
    PredictionContext,
)
from financial_volatility.models.base import ForecastModel


@dataclass(slots=True)
class LinearRegressionModel(ForecastModel):
    """Linear regression adapter for tabular volatility forecasting features."""

    _estimator: LinearRegression = field(
        default_factory=LinearRegression,
        init=False,
        repr=False,
    )
    _is_trained: bool = field(default=False, init=False, repr=False)

    def train(
        self,
        data: ModelInput,
        validation_data: ModelInput | None = None,
    ) -> None:
        """Fit linear regression on tabular features and volatility targets."""
        _ = validation_data
        features = _feature_matrix(data.features)
        target = _target_array(data.target)

        if len(features) != len(target):
            msg = (
                "LinearRegressionModel requires features and target with the "
                f"same length: {len(features)} != {len(target)}"
            )
            raise ValueError(msg)

        self._estimator.fit(features, target)
        self._is_trained = True

    def predict(self, context: PredictionContext, horizon: int) -> Forecast:
        """Predict realized volatility for the provided feature rows."""
        if not self._is_trained:
            raise ValueError("LinearRegressionModel must be trained before prediction")

        if horizon <= 0:
            raise ValueError("horizon must be positive")

        features = _feature_matrix(context.features)
        predictions = self._estimator.predict(features)

        if len(predictions) < horizon:
            msg = (
                "Prediction context does not contain enough feature rows for "
                f"horizon {horizon}: {len(predictions)} rows"
            )
            raise ValueError(msg)

        values = np.asarray(predictions[:horizon], dtype=np.float64)
        return Forecast(
            values=values.tolist(),
            horizon=horizon,
            timestamps=context.timestamps,
        )

    def save(self, path: str | Path) -> None:
        """Persist model state using joblib."""
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load a previously persisted model adapter."""
        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            msg = f"Persisted object is not a {cls.__name__}"
            raise TypeError(msg)
        return loaded

    def metadata(self) -> ModelMetadata:
        """Return reproducibility metadata for benchmark result records."""
        return ModelMetadata(
            name="linear_regression",
            model_family="statistical_baseline",
            parameters={},
            supported_hardware=(HardwareTarget.CPU,),
        )


def _feature_matrix(features: object) -> npt.NDArray[np.float64]:
    """Convert model features to a two-dimensional numeric array."""
    if isinstance(features, pd.DataFrame):
        values = features.to_numpy(dtype=np.float64)
    else:
        values = np.asarray(cast(npt.ArrayLike, features), dtype=np.float64)

    if values.ndim == 1:
        values = values.reshape(-1, 1)

    if values.ndim != 2:
        raise ValueError("LinearRegressionModel features must be two-dimensional")

    if values.shape[0] == 0:
        raise ValueError("LinearRegressionModel features must be non-empty")

    if not np.all(np.isfinite(values)):
        raise ValueError("LinearRegressionModel features must be finite")

    return values


def _target_array(target: object) -> npt.NDArray[np.float64]:
    """Convert model targets to a one-dimensional numeric array."""
    values = np.asarray(cast(npt.ArrayLike, target), dtype=np.float64).reshape(-1)

    if values.size == 0:
        raise ValueError("LinearRegressionModel target must be non-empty")

    if not np.all(np.isfinite(values)):
        raise ValueError("LinearRegressionModel target must be finite")

    return values


__all__ = ["LinearRegressionModel"]
