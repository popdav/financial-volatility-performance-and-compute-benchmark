"""Feature engineering tests."""

import math

import pandas as pd
import pytest

from financial_volatility.data.types import OHLCVData
from financial_volatility.features.engineering import (
    add_lagged_returns,
    add_lagged_volatility,
    add_log_returns,
    add_moving_averages,
    add_realized_volatility,
    add_simple_returns,
    add_volume_change,
    build_volatility_features,
)


def test_simple_returns_are_generated_from_close_prices() -> None:
    """Simple returns use close-to-close percentage change."""
    features = add_simple_returns(_ohlcv_frame())

    assert features["return_1d"].iloc[1] == pytest.approx(0.01)
    assert features["return_1d"].iloc[2] == pytest.approx(102.0 / 101.0 - 1.0)


def test_log_returns_are_generated_from_close_prices() -> None:
    """Log returns use log close price ratios."""
    features = add_log_returns(_ohlcv_frame())

    assert features["log_return_1d"].iloc[1] == pytest.approx(math.log(101.0 / 100.0))


def test_rolling_volatility_is_calculated_from_log_returns() -> None:
    """Realized volatility is a rolling standard deviation of log returns."""
    frame = add_log_returns(_ohlcv_frame())

    features = add_realized_volatility(frame, windows=(5,))

    expected = frame["log_return_1d"].rolling(window=5).std().iloc[5]
    assert features["realized_volatility_5d"].iloc[5] == pytest.approx(expected)


def test_lagged_returns_are_shifted_correctly() -> None:
    """Lagged return features shift the simple return column."""
    frame = add_simple_returns(_ohlcv_frame())

    features = add_lagged_returns(frame, lags=(1, 5))

    assert features["return_lag_1"].iloc[2] == pytest.approx(
        frame["return_1d"].iloc[1],
    )
    assert features["return_lag_5"].iloc[6] == pytest.approx(
        frame["return_1d"].iloc[1],
    )


def test_lagged_volatility_is_shifted_correctly() -> None:
    """Lagged volatility features shift realized volatility."""
    frame = add_realized_volatility(_ohlcv_frame(), windows=(5,))

    features = add_lagged_volatility(frame, lags=(1,))

    assert features["volatility_lag_1"].iloc[6] == pytest.approx(
        frame["realized_volatility_5d"].iloc[5],
    )


def test_moving_averages_are_calculated_from_close_prices() -> None:
    """Moving averages are rolling means of close prices."""
    frame = _ohlcv_frame()

    features = add_moving_averages(frame, windows=(5, 21))

    assert features["ma_5"].iloc[4] == pytest.approx(frame["close"].iloc[:5].mean())
    assert features["ma_21"].iloc[20] == pytest.approx(frame["close"].iloc[:21].mean())


def test_volume_change_is_generated_from_volume() -> None:
    """Volume change uses one-period percentage change."""
    frame = _ohlcv_frame()

    features = add_volume_change(frame)

    assert features["volume_change"].iloc[1] == pytest.approx(1001 / 1000 - 1)


def test_default_feature_generation_works_on_ohlcv_data() -> None:
    """The default feature set accepts OHLCVData and returns engineered features."""
    data = OHLCVData(_ohlcv_frame())

    features = build_volatility_features(data)

    assert list(features.columns) == [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "return_1d",
        "log_return_1d",
        "realized_volatility_5d",
        "realized_volatility_21d",
        "return_lag_1",
        "return_lag_5",
        "volatility_lag_1",
        "ma_5",
        "ma_21",
        "volume_change",
    ]
    assert not features.isna().any().any()
    assert features.index[0] == pd.Timestamp("2026-01-22")


def test_default_feature_generation_drops_insufficient_history_rows() -> None:
    """Rolling-window and lag warm-up rows are dropped from the final output."""
    features = build_volatility_features(_ohlcv_frame())

    assert len(features) == 4
    assert features.index[0] == pd.Timestamp("2026-01-22")


def test_feature_functions_do_not_mutate_input_frames() -> None:
    """Feature engineering returns new DataFrames instead of mutating inputs."""
    frame = _ohlcv_frame()

    _ = build_volatility_features(frame)

    assert "return_1d" not in frame.columns
    assert "log_return_1d" not in frame.columns


def _ohlcv_frame() -> pd.DataFrame:
    """Create synthetic OHLCV data with enough history for default features."""
    dates = pd.date_range("2026-01-01", periods=25, freq="D")
    close = [100.0 + index for index in range(25)]
    return pd.DataFrame(
        {
            "open": close,
            "high": [price + 1.0 for price in close],
            "low": [price - 1.0 for price in close],
            "close": close,
            "volume": [1000 + index for index in range(25)],
        },
        index=dates,
    )
