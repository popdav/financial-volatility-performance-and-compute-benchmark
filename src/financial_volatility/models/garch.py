"""GARCH forecasting model adapter."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from arch import arch_model

from financial_volatility.benchmark.types import (
    Forecast,
    HardwareTarget,
    ModelInput,
    ModelMetadata,
    PredictionContext,
)
from financial_volatility.models.base import ForecastModel


@dataclass(slots=True)
class GARCHModel(ForecastModel):
    """GARCH(p, q) volatility forecasting adapter.

    The default configuration is GARCH(1, 1). The adapter trains on log returns
    when a ``log_return_1d`` feature column is available, otherwise it falls
    back to the supplied target series.
    """

    p: int = 1
    q: int = 1
    return_column: str = "log_return_1d"
    _result: Any = field(default=None, init=False, repr=False)
    _return_scale: float = field(default=100.0, init=False, repr=False)

    def train(
        self,
        data: ModelInput,
        validation_data: ModelInput | None = None,
    ) -> None:
        """Fit a GARCH model to the supplied return series."""
        _ = validation_data
        returns = _extract_return_series(data, self.return_column)
        if returns.size < 20:
            msg = "GARCHModel requires at least 20 return observations for training"
            raise ValueError(msg)

        scaled_returns = returns * self._return_scale
        model = arch_model(
            scaled_returns,
            mean="Zero",
            vol="GARCH",
            p=self.p,
            q=self.q,
            rescale=False,
        )
        self._result = model.fit(disp="off")

    def predict(self, context: PredictionContext, horizon: int) -> Forecast:
        """Forecast one or more steps of conditional volatility."""
        if self._result is None:
            raise ValueError("GARCHModel must be trained before prediction")

        if horizon <= 0:
            raise ValueError("horizon must be positive")

        forecast = self._result.forecast(horizon=horizon, reindex=False)
        variances = np.asarray(forecast.variance.iloc[-1].to_numpy(), dtype=np.float64)
        volatility = np.sqrt(np.maximum(variances, 0.0)) / self._return_scale
        return Forecast(
            values=volatility.tolist(),
            horizon=horizon,
            timestamps=context.timestamps,
        )

    def save(self, path: str | Path) -> None:
        """Persist fitted model state."""
        with Path(path).open("wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load a previously persisted GARCH adapter."""
        with Path(path).open("rb") as file:
            loaded = pickle.load(file)

        if not isinstance(loaded, cls):
            msg = f"Persisted object is not a {cls.__name__}"
            raise TypeError(msg)
        return loaded

    def metadata(self) -> ModelMetadata:
        """Return reproducibility metadata for benchmark result records."""
        return ModelMetadata(
            name="garch",
            model_family="statistical",
            parameters={"p": self.p, "q": self.q},
            supported_hardware=(HardwareTarget.CPU,),
        )


def _extract_return_series(
    data: ModelInput,
    return_column: str,
) -> npt.NDArray[np.float64]:
    """Extract a finite numeric return series from model input."""
    if (
        isinstance(data.features, pd.DataFrame)
        and return_column in data.features.columns
    ):
        raw_values: object = data.features[return_column]
    else:
        raw_values = data.target

    values = np.asarray(cast(npt.ArrayLike, raw_values), dtype=np.float64)
    values = values.reshape(-1)
    values = values[np.isfinite(values)]

    if values.size == 0:
        raise ValueError("GARCHModel training data must contain finite returns")

    return values


__all__ = ["GARCHModel"]
