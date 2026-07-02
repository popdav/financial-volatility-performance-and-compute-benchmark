"""Market data type tests."""

import pandas as pd
import pytest

from financial_volatility.data.types import (
    OHLCV_COLUMNS,
    OHLCVData,
    TimeSeriesDataset,
    TrainTestSplit,
)


def test_valid_ohlcv_data_is_accepted() -> None:
    """A non-empty chronological DataFrame with OHLCV columns is valid."""
    data = OHLCVData(
        data=_ohlcv_frame(),
        symbol="TEST",
        provider="csv",
    )

    assert data.symbol == "TEST"
    assert data.provider == "csv"
    assert data.columns == OHLCV_COLUMNS
    assert data.row_count == 3
    assert data.start_timestamp == pd.Timestamp("2026-01-01")
    assert data.end_timestamp == pd.Timestamp("2026-01-03")


def test_empty_data_raises_value_error() -> None:
    """OHLCV data must contain at least one row."""
    empty = _ohlcv_frame().iloc[0:0]

    with pytest.raises(ValueError, match="non-empty"):
        OHLCVData(empty)


def test_missing_required_columns_raise_value_error() -> None:
    """OHLCV data requires open, high, low, close, and volume columns."""
    missing_volume = _ohlcv_frame().drop(columns=["volume"])

    with pytest.raises(ValueError, match="missing required columns: volume"):
        OHLCVData(missing_volume)


def test_non_datetime_index_raises_value_error() -> None:
    """The DataFrame index must be a DatetimeIndex."""
    frame = _ohlcv_frame().reset_index(drop=True)

    with pytest.raises(ValueError, match="DatetimeIndex"):
        OHLCVData(frame)


def test_unsorted_dates_raise_value_error() -> None:
    """OHLCV rows must be sorted oldest to newest."""
    unsorted = _ohlcv_frame().sort_index(ascending=False)

    with pytest.raises(ValueError, match="chronological order"):
        OHLCVData(unsorted)


def test_duplicate_dates_raise_value_error() -> None:
    """Duplicate timestamps are rejected because splits require unique ordering."""
    duplicate = _ohlcv_frame()
    duplicate.index = pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-03"])

    with pytest.raises(ValueError, match="duplicate timestamps"):
        OHLCVData(duplicate)


def test_time_series_dataset_validates_required_columns() -> None:
    """Generic datasets can specify their own required model-facing columns."""
    dataset = TimeSeriesDataset(
        data=_ohlcv_frame(),
        name="close-prices",
        required_columns=("close",),
    )

    assert dataset.name == "close-prices"
    assert dataset.columns == OHLCV_COLUMNS
    assert dataset.start_timestamp == pd.Timestamp("2026-01-01")


def test_time_series_dataset_rejects_non_datetime_index() -> None:
    """Generic time-series datasets also use the index as the time axis."""
    frame = _ohlcv_frame().reset_index(drop=True)

    with pytest.raises(ValueError, match="DatetimeIndex"):
        TimeSeriesDataset(data=frame, name="missing-time")


def test_train_test_split_stores_train_and_test_datasets_correctly() -> None:
    """A split keeps the provided train and test datasets."""
    train = TimeSeriesDataset(data=_ohlcv_frame().iloc[:2], name="train")
    test = TimeSeriesDataset(data=_ohlcv_frame().iloc[2:], name="test")

    split = TrainTestSplit(train=train, test=test, name="holdout")

    assert split.name == "holdout"
    assert split.train is train
    assert split.test is test
    assert split.train.end_timestamp == pd.Timestamp("2026-01-02")
    assert split.test.start_timestamp == pd.Timestamp("2026-01-03")


def test_train_test_split_rejects_overlapping_datasets() -> None:
    """Train data must end before test data begins."""
    train = TimeSeriesDataset(data=_ohlcv_frame().iloc[:2], name="train")
    test = TimeSeriesDataset(data=_ohlcv_frame().iloc[1:], name="test")

    with pytest.raises(ValueError, match="non-overlapping"):
        TrainTestSplit(train=train, test=test)


def test_dataframe_access_returns_defensive_copy() -> None:
    """Mutating inputs or returned frames does not mutate stored data."""
    frame = _ohlcv_frame()
    data = OHLCVData(frame)

    frame.loc[pd.Timestamp("2026-01-01"), "close"] = 999.0
    returned = data.to_dataframe()
    returned.loc[pd.Timestamp("2026-01-02"), "close"] = 888.0

    stored = data.to_dataframe()
    assert stored.loc[pd.Timestamp("2026-01-01"), "close"] == 1.5
    assert stored.loc[pd.Timestamp("2026-01-02"), "close"] == 2.5


def test_ohlcv_data_converts_to_time_series_dataset() -> None:
    """OHLCV data can be exposed through the generic dataset abstraction."""
    data = OHLCVData(_ohlcv_frame())

    dataset = data.to_time_series_dataset(name="ohlcv")

    assert dataset.name == "ohlcv"
    assert dataset.columns == OHLCV_COLUMNS
    assert dataset.row_count == 3


def _ohlcv_frame() -> pd.DataFrame:
    """Create a minimal valid OHLCV frame."""
    return pd.DataFrame(
        {
            "open": [1.0, 2.0, 3.0],
            "high": [2.0, 3.0, 4.0],
            "low": [0.5, 1.5, 2.5],
            "close": [1.5, 2.5, 3.5],
            "volume": [100, 200, 300],
        },
        index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
    )
