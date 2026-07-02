"""Time-series train/test splitting helpers."""

from __future__ import annotations

from datetime import date

import pandas as pd

from financial_volatility.data.types import TimeSeriesDataset, TrainTestSplit


def split_time_series(
    data: pd.DataFrame | TimeSeriesDataset,
    *,
    test_size: float | None = None,
    split_date: str | date | pd.Timestamp | None = None,
    name: str = "default",
) -> TrainTestSplit:
    """Split time-indexed data into chronological train and test datasets."""
    if (test_size is None) == (split_date is None):
        msg = "Provide exactly one of test_size or split_date"
        raise ValueError(msg)

    frame = _as_sorted_dataframe(data)
    if frame.empty:
        raise ValueError("Input data must be non-empty")

    if test_size is not None:
        train_frame, test_frame = _split_by_test_size(frame, test_size)
    else:
        train_frame, test_frame = _split_by_date(frame, split_date)

    _validate_non_empty_split(train_frame, test_frame)
    return TrainTestSplit(
        train=TimeSeriesDataset(data=train_frame, name=f"{name}_train"),
        test=TimeSeriesDataset(data=test_frame, name=f"{name}_test"),
        name=name,
    )


def _as_sorted_dataframe(data: pd.DataFrame | TimeSeriesDataset) -> pd.DataFrame:
    """Return a chronological DataFrame copy from supported split inputs."""
    if isinstance(data, TimeSeriesDataset):
        frame = data.to_dataframe()
    else:
        frame = data.copy(deep=True)

    if not isinstance(frame.index, pd.DatetimeIndex):
        msg = "Input data index must be a pandas DatetimeIndex"
        raise ValueError(msg)

    return frame.sort_index()


def _split_by_test_size(
    frame: pd.DataFrame,
    test_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data by fractional test size."""
    if test_size <= 0.0 or test_size >= 1.0:
        raise ValueError("test_size must be greater than 0 and less than 1")

    test_count = int(len(frame) * test_size)
    if test_count == 0:
        msg = "test_size produces an empty test set"
        raise ValueError(msg)

    split_index = len(frame) - test_count
    return frame.iloc[:split_index], frame.iloc[split_index:]


def _split_by_date(
    frame: pd.DataFrame,
    split_date: str | date | pd.Timestamp | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data at a date boundary, with the date included in test."""
    if split_date is None:
        raise ValueError("split_date must be provided")

    try:
        split_timestamp = pd.Timestamp(split_date)
    except (TypeError, ValueError) as error:
        msg = f"split_date must be parseable as a timestamp: {split_date!r}"
        raise ValueError(msg) from error

    train_frame = frame.loc[frame.index < split_timestamp]
    test_frame = frame.loc[frame.index >= split_timestamp]
    return train_frame, test_frame


def _validate_non_empty_split(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> None:
    """Validate that both sides of the split contain rows."""
    if train_frame.empty:
        raise ValueError("Split produces an empty train set")

    if test_frame.empty:
        raise ValueError("Split produces an empty test set")


__all__ = ["split_time_series"]
