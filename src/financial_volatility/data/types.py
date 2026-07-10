"""Provider-agnostic market data containers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import InitVar, dataclass, field
from typing import cast

import numpy as np
import pandas as pd

OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


class MarketDataValidationError(ValueError):
    """Raised when market data does not satisfy the benchmark data contract."""


@dataclass(frozen=True, slots=True)
class OHLCVData:
    """Immutable OHLCV market data backed by a pandas DataFrame."""

    data: InitVar[pd.DataFrame]
    symbol: str | None = None
    provider: str | None = None
    _data: pd.DataFrame = field(init=False, repr=False, compare=False)

    def __post_init__(self, data: pd.DataFrame) -> None:
        """Validate and store a defensive copy of OHLCV data."""
        validated = _validated_frame(
            data,
            required_columns=OHLCV_COLUMNS,
            label="OHLCV data",
        )
        object.__setattr__(self, "_data", validated)

    @property
    def columns(self) -> tuple[str, ...]:
        """Return the DataFrame column names."""
        return tuple(str(column) for column in self._data.columns)

    @property
    def start_timestamp(self) -> pd.Timestamp:
        """Return the first timestamp in the data."""
        return _datetime_index(self._data)[0]

    @property
    def end_timestamp(self) -> pd.Timestamp:
        """Return the final timestamp in the data."""
        return _datetime_index(self._data)[-1]

    @property
    def row_count(self) -> int:
        """Return the number of observations."""
        return len(self._data)

    def to_dataframe(self) -> pd.DataFrame:
        """Return a defensive copy of the underlying DataFrame."""
        return self._data.copy(deep=True)

    def to_time_series_dataset(self, name: str) -> TimeSeriesDataset:
        """Represent OHLCV data as a generic model-facing dataset."""
        return TimeSeriesDataset(
            data=self._data,
            name=name,
            required_columns=OHLCV_COLUMNS,
        )


OHLCVMarketData = OHLCVData


@dataclass(frozen=True, slots=True)
class TimeSeriesDataset:
    """Immutable model-facing time-series dataset backed by a DataFrame."""

    data: InitVar[pd.DataFrame]
    name: str
    required_columns: Sequence[str] = ()
    _data: pd.DataFrame = field(init=False, repr=False, compare=False)

    def __post_init__(self, data: pd.DataFrame) -> None:
        """Validate and store a defensive copy of time-series data."""
        validated = _validated_frame(
            data,
            required_columns=self.required_columns,
            label=f"time-series dataset {self.name!r}",
        )
        object.__setattr__(self, "_data", validated)

    @property
    def columns(self) -> tuple[str, ...]:
        """Return the DataFrame column names."""
        return tuple(str(column) for column in self._data.columns)

    @property
    def start_timestamp(self) -> pd.Timestamp:
        """Return the first timestamp in the data."""
        return _datetime_index(self._data)[0]

    @property
    def end_timestamp(self) -> pd.Timestamp:
        """Return the final timestamp in the data."""
        return _datetime_index(self._data)[-1]

    @property
    def row_count(self) -> int:
        """Return the number of observations."""
        return len(self._data)

    def to_dataframe(self) -> pd.DataFrame:
        """Return a defensive copy of the underlying DataFrame."""
        return self._data.copy(deep=True)


@dataclass(frozen=True, slots=True)
class TrainTestSplit:
    """Immutable chronological train/test split."""

    train: TimeSeriesDataset
    test: TimeSeriesDataset
    name: str = "default"

    def __post_init__(self) -> None:
        """Validate that train data ends before test data begins."""
        if self.train.end_timestamp >= self.test.start_timestamp:
            msg = (
                "Train/test split must be chronological and non-overlapping: "
                f"train ends at {self.train.end_timestamp}, "
                f"test starts at {self.test.start_timestamp}"
            )
            raise MarketDataValidationError(msg)


def _validated_frame(
    data: pd.DataFrame,
    *,
    required_columns: Sequence[str],
    label: str,
) -> pd.DataFrame:
    """Validate tabular time-series data and return a defensive copy."""
    if not isinstance(data, pd.DataFrame):
        msg = f"{label} must be a pandas DataFrame"
        raise MarketDataValidationError(msg)

    if data.empty:
        msg = f"{label} must be non-empty"
        raise MarketDataValidationError(msg)

    _validate_required_columns(data, required_columns, label)
    _validate_datetime_index(data, label)
    _validate_chronological_order(_datetime_index(data), label)
    _validate_required_values(data, required_columns, label)

    return data.copy(deep=True)


def _validate_required_columns(
    data: pd.DataFrame,
    required_columns: Sequence[str],
    label: str,
) -> None:
    """Validate that all required columns are present."""
    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        msg = f"{label} is missing required columns: {', '.join(missing)}"
        raise MarketDataValidationError(msg)


def _validate_datetime_index(data: pd.DataFrame, label: str) -> None:
    """Validate that the DataFrame index is the time axis."""
    if not isinstance(data.index, pd.DatetimeIndex):
        msg = f"{label} index must be a pandas DatetimeIndex"
        raise MarketDataValidationError(msg)


def _datetime_index(data: pd.DataFrame) -> pd.DatetimeIndex:
    """Return a DataFrame index that has already been validated as datetime."""
    return cast(pd.DatetimeIndex, data.index)


def _validate_chronological_order(
    timestamps: pd.DatetimeIndex,
    label: str,
) -> None:
    """Validate timestamps are ordered and unique."""
    if timestamps.hasnans:
        msg = f"{label} index contains invalid or missing timestamps"
        raise MarketDataValidationError(msg)

    if not timestamps.is_monotonic_increasing:
        msg = f"{label} must be sorted in chronological order"
        raise MarketDataValidationError(msg)

    if not timestamps.is_unique:
        msg = f"{label} index contains duplicate timestamps"
        raise MarketDataValidationError(msg)


def _validate_required_values(
    data: pd.DataFrame,
    required_columns: Sequence[str],
    label: str,
) -> None:
    """Validate required columns contain finite numeric values."""
    for column in required_columns:
        values = pd.to_numeric(data[column], errors="coerce")
        if values.isna().any():
            msg = f"{label} column {column!r} contains missing or non-numeric values"
            raise MarketDataValidationError(msg)

        if not np.all(np.isfinite(values.to_numpy(dtype=np.float64))):
            msg = f"{label} column {column!r} contains non-finite values"
            raise MarketDataValidationError(msg)


__all__ = [
    "OHLCV_COLUMNS",
    "MarketDataValidationError",
    "OHLCVData",
    "OHLCVMarketData",
    "TimeSeriesDataset",
    "TrainTestSplit",
]
