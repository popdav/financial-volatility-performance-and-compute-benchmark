"""Time-series splitting tests."""

from datetime import date

import pandas as pd
import pytest

from financial_volatility.data.splitting import split_time_series
from financial_volatility.data.types import TimeSeriesDataset, TrainTestSplit


def test_float_based_split_works() -> None:
    """A fractional test size reserves the final chronological observations."""
    split = split_time_series(_frame(), test_size=0.2, name="holdout")

    assert isinstance(split, TrainTestSplit)
    assert split.name == "holdout"
    assert split.train.row_count == 8
    assert split.test.row_count == 2
    assert split.train.end_timestamp == pd.Timestamp("2026-01-08")
    assert split.test.start_timestamp == pd.Timestamp("2026-01-09")


def test_date_based_split_works_with_string() -> None:
    """A split date becomes the first date in the test set."""
    split = split_time_series(_frame(), split_date="2026-01-07")

    assert split.train.row_count == 6
    assert split.test.row_count == 4
    assert split.train.end_timestamp == pd.Timestamp("2026-01-06")
    assert split.test.start_timestamp == pd.Timestamp("2026-01-07")


def test_date_based_split_works_with_date_object() -> None:
    """Date objects are accepted as split boundaries."""
    split = split_time_series(_frame(), split_date=date(2026, 1, 7))

    assert split.test.start_timestamp == pd.Timestamp("2026-01-07")


def test_invalid_test_size_raises_value_error() -> None:
    """test_size must be a fraction between zero and one."""
    with pytest.raises(ValueError, match="greater than 0 and less than 1"):
        split_time_series(_frame(), test_size=1.0)


def test_split_producing_empty_test_raises_value_error() -> None:
    """Small fractional test sizes cannot produce an empty test set."""
    with pytest.raises(ValueError, match="empty test set"):
        split_time_series(_frame(rows=3), test_size=0.2)


def test_split_producing_empty_train_raises_value_error() -> None:
    """Date boundaries before all observations produce an empty train set."""
    with pytest.raises(ValueError, match="empty train set"):
        split_time_series(_frame(), split_date="2026-01-01")


def test_empty_input_raises_value_error() -> None:
    """Input data must contain at least one row."""
    with pytest.raises(ValueError, match="non-empty"):
        split_time_series(_frame(rows=0), test_size=0.2)


def test_output_remains_chronologically_sorted() -> None:
    """Input rows are sorted before splitting and output order is chronological."""
    unsorted = _frame().sort_index(ascending=False)

    split = split_time_series(unsorted, test_size=0.3)

    assert split.train.to_dataframe().index.is_monotonic_increasing
    assert split.test.to_dataframe().index.is_monotonic_increasing
    assert split.train.end_timestamp < split.test.start_timestamp


def test_time_series_dataset_input_is_supported() -> None:
    """Existing TimeSeriesDataset instances can be split directly."""
    dataset = TimeSeriesDataset(data=_frame(), name="features")

    split = split_time_series(dataset, test_size=0.2)

    assert split.train.row_count == 8
    assert split.test.row_count == 2


def test_non_datetime_index_raises_value_error() -> None:
    """Splitting requires a DatetimeIndex."""
    frame = _frame().reset_index(drop=True)

    with pytest.raises(ValueError, match="DatetimeIndex"):
        split_time_series(frame, test_size=0.2)


def test_requires_exactly_one_split_strategy() -> None:
    """The caller must choose either test_size or split_date."""
    with pytest.raises(ValueError, match="exactly one"):
        split_time_series(_frame())

    with pytest.raises(ValueError, match="exactly one"):
        split_time_series(_frame(), test_size=0.2, split_date="2026-01-07")


def _frame(rows: int = 10) -> pd.DataFrame:
    """Create synthetic time-indexed data."""
    index = pd.date_range("2026-01-01", periods=rows, freq="D")
    return pd.DataFrame({"value": list(range(rows))}, index=index)
