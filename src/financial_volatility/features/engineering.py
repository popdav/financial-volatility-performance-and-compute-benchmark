"""Feature engineering primitives for volatility forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from financial_volatility.data.types import OHLCVData


def add_simple_returns(data: pd.DataFrame | OHLCVData) -> pd.DataFrame:
    """Add one-period simple returns calculated from close prices."""
    frame = _as_dataframe(data)
    frame["return_1d"] = frame["close"].pct_change()
    return frame


def add_log_returns(data: pd.DataFrame | OHLCVData) -> pd.DataFrame:
    """Add one-period log returns calculated from close prices."""
    frame = _as_dataframe(data)
    frame["log_return_1d"] = np.log(frame["close"] / frame["close"].shift(1))
    return frame


def add_realized_volatility(
    data: pd.DataFrame | OHLCVData,
    *,
    windows: tuple[int, ...] = (5, 21),
) -> pd.DataFrame:
    """Add rolling realized volatility features from log returns."""
    frame = _ensure_log_returns(_as_dataframe(data))
    for window in windows:
        frame[f"realized_volatility_{window}d"] = (
            frame["log_return_1d"].rolling(window=window).std()
        )
    return frame


def add_lagged_returns(
    data: pd.DataFrame | OHLCVData,
    *,
    lags: tuple[int, ...] = (1, 5),
) -> pd.DataFrame:
    """Add shifted simple return features."""
    frame = _ensure_simple_returns(_as_dataframe(data))
    for lag in lags:
        frame[f"return_lag_{lag}"] = frame["return_1d"].shift(lag)
    return frame


def add_lagged_volatility(
    data: pd.DataFrame | OHLCVData,
    *,
    volatility_column: str = "realized_volatility_5d",
    lags: tuple[int, ...] = (1,),
) -> pd.DataFrame:
    """Add shifted realized volatility features."""
    frame = _as_dataframe(data)
    if volatility_column not in frame.columns:
        frame = add_realized_volatility(frame, windows=(5,))
    for lag in lags:
        frame[f"volatility_lag_{lag}"] = frame[volatility_column].shift(lag)
    return frame


def add_moving_averages(
    data: pd.DataFrame | OHLCVData,
    *,
    windows: tuple[int, ...] = (5, 21),
) -> pd.DataFrame:
    """Add rolling moving averages calculated from close prices."""
    frame = _as_dataframe(data)
    for window in windows:
        frame[f"ma_{window}"] = frame["close"].rolling(window=window).mean()
    return frame


def build_volatility_features(data: pd.DataFrame | OHLCVData) -> pd.DataFrame:
    """Build the default volatility forecasting feature set."""
    frame = add_simple_returns(data)
    frame = add_log_returns(frame)
    frame = add_realized_volatility(frame, windows=(5, 21))
    frame = add_lagged_returns(frame, lags=(1, 5))
    frame = add_lagged_volatility(
        frame,
        volatility_column="realized_volatility_5d",
        lags=(1,),
    )
    frame = add_moving_averages(frame, windows=(5, 21))
    return frame.dropna()


def _as_dataframe(data: pd.DataFrame | OHLCVData) -> pd.DataFrame:
    """Return a defensive DataFrame copy from supported feature inputs."""
    if isinstance(data, OHLCVData):
        return data.to_dataframe()
    return data.copy(deep=True)


def _ensure_simple_returns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add simple returns when they are not already present."""
    if "return_1d" not in frame.columns:
        frame["return_1d"] = frame["close"].pct_change()
    return frame


def _ensure_log_returns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add log returns when they are not already present."""
    if "log_return_1d" not in frame.columns:
        frame["log_return_1d"] = np.log(frame["close"] / frame["close"].shift(1))
    return frame
