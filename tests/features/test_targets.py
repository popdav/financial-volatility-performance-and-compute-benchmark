"""Target construction and supervised dataset tests."""

import numpy as np
import pandas as pd
import pytest

from financial_volatility.data.types import OHLCVData
from financial_volatility.features.engineering import (
    build_supervised_dataset,
    make_volatility_target,
)


def test_make_volatility_target_horizon_1_uses_next_return() -> None:
    """A one-step target uses only the next log return."""
    frame = _return_frame()

    target = make_volatility_target(frame, horizon=1)

    assert target.iloc[0] == pytest.approx(abs(frame["log_return_1d"].iloc[1]))
    assert pd.isna(target.iloc[-1])


def test_make_volatility_target_horizon_5_uses_future_returns() -> None:
    """Multi-step targets aggregate only future returns."""
    frame = _return_frame()

    target = make_volatility_target(frame, horizon=5)

    expected = np.sqrt(np.mean(np.square(frame["log_return_1d"].iloc[1:6])))
    assert target.iloc[0] == pytest.approx(expected)


def test_make_volatility_target_horizon_21_uses_future_returns() -> None:
    """The monthly thesis horizon aggregates the next 21 returns."""
    frame = _return_frame(rows=40)

    target = make_volatility_target(frame, horizon=21)

    expected = np.sqrt(np.mean(np.square(frame["log_return_1d"].iloc[1:22])))
    assert target.iloc[0] == pytest.approx(expected)


def test_make_volatility_target_rejects_invalid_horizon() -> None:
    """Forecast horizons must be positive."""
    with pytest.raises(ValueError, match="horizon"):
        make_volatility_target(_return_frame(), horizon=0)


def test_build_supervised_dataset_returns_aligned_clean_X_y() -> None:
    """The supervised builder returns equal-length clean features and targets."""
    X, y = build_supervised_dataset(
        OHLCVData(_ohlcv_frame()),
        horizon=5,
        volatility_windows=(5,),
        return_lags=(1,),
        volatility_lags=(1,),
        moving_average_windows=(5,),
    )

    assert len(X) == len(y)
    assert len(X) > 0
    assert isinstance(X.index, pd.DatetimeIndex)
    assert X.index.equals(y.index)
    assert not X.isna().any().any()
    assert not y.isna().any()
    assert y.name == "realized_volatility_target_5d"


def test_build_supervised_dataset_preserves_no_lookahead_alignment() -> None:
    """The first target remains based on returns after the feature timestamp."""
    X, y = build_supervised_dataset(
        _ohlcv_frame(rows=30),
        horizon=5,
        volatility_windows=(5,),
        return_lags=(1,),
        volatility_lags=(1,),
        moving_average_windows=(5,),
    )
    _ = X
    feature_timestamp = y.index[0]
    frame = _return_frame(rows=30)
    position = frame.index.get_loc(feature_timestamp)

    expected = np.sqrt(
        np.mean(np.square(frame["log_return_1d"].iloc[position + 1 : position + 6]))
    )
    assert y.iloc[0] == pytest.approx(expected)


def _return_frame(rows: int = 30) -> pd.DataFrame:
    """Create deterministic log returns."""
    index = pd.date_range("2026-01-01", periods=rows, freq="D")
    close = pd.Series(np.linspace(100.0, 130.0, num=rows), index=index)
    return pd.DataFrame(
        {
            "close": close,
            "log_return_1d": np.log(close / close.shift(1)),
        },
        index=index,
    )


def _ohlcv_frame(rows: int = 40) -> pd.DataFrame:
    """Create synthetic OHLCV data."""
    index = pd.date_range("2026-01-01", periods=rows, freq="D")
    close = np.linspace(100.0, 130.0, num=rows)
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.linspace(1000.0, 1500.0, num=rows),
        },
        index=index,
    )
