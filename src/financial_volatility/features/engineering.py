"""Leakage-safe feature engineering for volatility forecasting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from financial_volatility.data.types import OHLCVData

FeatureCategory = Literal["price", "trend", "volatility", "momentum", "volume"]
SUPPORTED_HORIZONS = (5, 21)
DEFAULT_VOLATILITY_WINDOWS = (5, 10, 21, 63)
DEFAULT_SMA_WINDOWS = (5, 10, 21, 50)
DEFAULT_EMA_WINDOWS = (10, 21, 50)


@dataclass(frozen=True, slots=True)
class FeatureMetadata:
    """Description of one deterministic model feature."""

    name: str
    category: FeatureCategory
    rolling_window: int | None
    description: str


@dataclass(frozen=True, slots=True)
class FeatureConfig:
    """Resolved feature selection loaded directly or from a YAML mapping."""

    price_enabled: bool = True
    volatility_windows: tuple[int, ...] = DEFAULT_VOLATILITY_WINDOWS
    sma_windows: tuple[int, ...] = DEFAULT_SMA_WINDOWS
    ema_windows: tuple[int, ...] = DEFAULT_EMA_WINDOWS
    rsi_enabled: bool = True
    rsi_period: int = 14
    atr_enabled: bool = True
    atr_period: int = 14
    volume_enabled: bool = True

    def __post_init__(self) -> None:
        """Reject nonsensical and duplicate rolling periods."""
        for label, windows in (
            ("volatility", self.volatility_windows),
            ("sma", self.sma_windows),
            ("ema", self.ema_windows),
        ):
            _validate_windows(label, windows)
        if self.rsi_period < 1:
            raise ValueError("rsi period must be at least 1")
        if self.atr_period < 1:
            raise ValueError("atr period must be at least 1")

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> FeatureConfig:
        """Resolve the ``features`` section of a YAML-compatible mapping."""
        root = config.get("features", config)
        if not isinstance(root, Mapping):
            raise ValueError("features configuration must be a mapping")

        def section(name: str) -> Mapping[str, Any]:
            value = root.get(name, {})
            if not isinstance(value, Mapping):
                raise ValueError(f"features.{name} must be a mapping")
            return value

        price = section("price")
        volatility = section("volatility")
        sma = section("sma")
        ema = section("ema")
        rsi = section("rsi")
        atr = section("atr")
        volume = section("volume")
        return cls(
            price_enabled=_bool_value(price, "enabled", True),
            volatility_windows=_windows_value(
                volatility, "windows", DEFAULT_VOLATILITY_WINDOWS
            ),
            sma_windows=_windows_value(sma, "windows", DEFAULT_SMA_WINDOWS),
            ema_windows=_windows_value(ema, "windows", DEFAULT_EMA_WINDOWS),
            rsi_enabled=_bool_value(rsi, "enabled", True),
            rsi_period=_int_value(rsi, "period", 14),
            atr_enabled=_bool_value(atr, "enabled", True),
            atr_period=_int_value(atr, "period", 14),
            volume_enabled=_bool_value(volume, "enabled", True),
        )


@dataclass(frozen=True, slots=True)
class DatasetValidationReport:
    """Validation evidence for a cleaned supervised dataset."""

    input_rows: int
    output_rows: int
    removed_rows: int
    feature_count: int


def add_simple_returns(data: pd.DataFrame | OHLCVData) -> pd.DataFrame:
    """Add adjusted-close one-period simple returns."""
    frame = _as_dataframe(data)
    _require_columns(frame, ("adjusted_close",))
    frame["simple_return"] = frame["adjusted_close"].pct_change(fill_method=None)
    return frame


def add_log_returns(data: pd.DataFrame | OHLCVData) -> pd.DataFrame:
    """Add adjusted-close log returns, preserving the first value as NaN."""
    frame = _as_dataframe(data)
    _require_columns(frame, ("adjusted_close",))
    frame["log_return"] = np.log(
        frame["adjusted_close"] / frame["adjusted_close"].shift(1)
    )
    return frame


def add_realized_volatility(
    data: pd.DataFrame | OHLCVData,
    *,
    windows: tuple[int, ...] = (5, 10, 21, 63),
) -> pd.DataFrame:
    """Add annualized historical sample volatility of adjusted log returns."""
    _validate_windows("volatility", windows)
    frame = _ensure_log_returns(_as_dataframe(data))
    for window in windows:
        frame[f"historical_volatility_{window}"] = frame["log_return"].rolling(
            window=window, min_periods=window
        ).std() * np.sqrt(252.0)
    return frame


def add_moving_averages(
    data: pd.DataFrame | OHLCVData,
    *,
    windows: tuple[int, ...] = (5, 10, 21, 50),
) -> pd.DataFrame:
    """Add adjusted-close simple moving averages."""
    _validate_windows("sma", windows)
    frame = _as_dataframe(data)
    _require_columns(frame, ("adjusted_close",))
    for window in windows:
        frame[f"sma_{window}"] = (
            frame["adjusted_close"].rolling(window, min_periods=window).mean()
        )
    return frame


def add_exponential_moving_averages(
    data: pd.DataFrame | OHLCVData,
    *,
    windows: tuple[int, ...] = (10, 21, 50),
) -> pd.DataFrame:
    """Add adjusted-close EMAs using span and non-adjusted recursion."""
    _validate_windows("ema", windows)
    frame = _as_dataframe(data)
    _require_columns(frame, ("adjusted_close",))
    for window in windows:
        frame[f"ema_{window}"] = (
            frame["adjusted_close"]
            .ewm(span=window, adjust=False, min_periods=window)
            .mean()
        )
    return frame


def add_rsi(data: pd.DataFrame | OHLCVData, *, period: int = 14) -> pd.DataFrame:
    """Add Wilder-smoothed RSI from adjusted-close changes."""
    if period < 1:
        raise ValueError("rsi period must be at least 1")
    frame = _as_dataframe(data)
    _require_columns(frame, ("adjusted_close",))
    delta = frame["adjusted_close"].diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    average_gain = _wilder_average(gains, period)
    average_loss = _wilder_average(losses, period)
    relative_strength = average_gain / average_loss
    rsi = 100.0 - 100.0 / (1.0 + relative_strength)
    rsi = rsi.mask((average_loss == 0) & (average_gain > 0), 100.0)
    rsi = rsi.mask((average_loss == 0) & (average_gain == 0), 50.0)
    frame[f"rsi_{period}"] = rsi
    return frame


def add_atr(data: pd.DataFrame | OHLCVData, *, period: int = 14) -> pd.DataFrame:
    """Add Wilder-smoothed average true range from raw High, Low, Close."""
    if period < 1:
        raise ValueError("atr period must be at least 1")
    frame = _as_dataframe(data)
    _require_columns(frame, ("high", "low", "close"))
    previous_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    frame[f"atr_{period}"] = _wilder_average(true_range, period)
    return frame


def add_volume_features(data: pd.DataFrame | OHLCVData) -> pd.DataFrame:
    """Add natural log volume and one-period percentage volume change."""
    frame = _as_dataframe(data)
    _require_columns(frame, ("volume",))
    with np.errstate(divide="ignore", invalid="ignore"):
        frame["log_volume"] = np.log(frame["volume"].to_numpy(dtype=np.float64))
    frame["volume_change"] = frame["volume"].pct_change(fill_method=None)
    return frame


def add_volume_change(data: pd.DataFrame | OHLCVData) -> pd.DataFrame:
    """Compatibility helper adding percentage volume change only."""
    frame = add_volume_features(data)
    return frame.drop(columns="log_volume")


def build_feature_matrix(
    data: pd.DataFrame | OHLCVData,
    *,
    config: FeatureConfig | Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, tuple[FeatureMetadata, ...]]:
    """Build configured, uncleaned features using information through time t."""
    resolved = _resolve_config(config)
    source = _as_dataframe(data)
    features = pd.DataFrame(index=source.index)
    metadata: list[FeatureMetadata] = []

    if resolved.price_enabled:
        priced = add_simple_returns(add_log_returns(source))
        features["log_return"] = priced["log_return"]
        features["simple_return"] = priced["simple_return"]
        metadata.extend(
            [
                FeatureMetadata(
                    "log_return", "price", 1, "Daily adjusted-price log return."
                ),
                FeatureMetadata(
                    "simple_return", "price", 1, "Daily adjusted-price simple return."
                ),
            ]
        )
    if resolved.volatility_windows:
        volatility = add_realized_volatility(
            source, windows=resolved.volatility_windows
        )
        for window in resolved.volatility_windows:
            name = f"historical_volatility_{window}"
            features[name] = volatility[name]
            metadata.append(
                FeatureMetadata(
                    name,
                    "volatility",
                    window,
                    f"Annualized {window}-day standard deviation of log returns.",
                )
            )
    if resolved.sma_windows:
        averages = add_moving_averages(source, windows=resolved.sma_windows)
        for window in resolved.sma_windows:
            name = f"sma_{window}"
            features[name] = averages[name]
            metadata.append(
                FeatureMetadata(
                    name, "trend", window, f"{window}-day simple moving average."
                )
            )
    if resolved.ema_windows:
        averages = add_exponential_moving_averages(source, windows=resolved.ema_windows)
        for window in resolved.ema_windows:
            name = f"ema_{window}"
            features[name] = averages[name]
            metadata.append(
                FeatureMetadata(
                    name, "trend", window, f"{window}-day exponential moving average."
                )
            )
    if resolved.rsi_enabled:
        momentum = add_rsi(source, period=resolved.rsi_period)
        name = f"rsi_{resolved.rsi_period}"
        features[name] = momentum[name]
        metadata.append(
            FeatureMetadata(
                name, "momentum", resolved.rsi_period, "Wilder relative strength index."
            )
        )
    if resolved.atr_enabled:
        volatility_range = add_atr(source, period=resolved.atr_period)
        name = f"atr_{resolved.atr_period}"
        features[name] = volatility_range[name]
        metadata.append(
            FeatureMetadata(
                name, "volatility", resolved.atr_period, "Wilder average true range."
            )
        )
    if resolved.volume_enabled:
        volume = add_volume_features(source)
        features[["log_volume", "volume_change"]] = volume[
            ["log_volume", "volume_change"]
        ]
        metadata.extend(
            [
                FeatureMetadata(
                    "log_volume", "volume", None, "Natural logarithm of daily volume."
                ),
                FeatureMetadata(
                    "volume_change", "volume", 1, "Daily percentage volume change."
                ),
            ]
        )
    return features, tuple(metadata)


def make_volatility_target(
    data: pd.DataFrame | OHLCVData,
    *,
    horizon: int,
    log_return_column: str = "log_return",
    name: str | None = None,
    annualize: bool = True,
) -> pd.Series:
    """Calculate RV(t,h) from exactly r(t+1), ..., r(t+h)."""
    if horizon < 2:
        raise ValueError("horizon must be at least 2")
    if horizon not in SUPPORTED_HORIZONS:
        raise ValueError(f"supported horizons are {SUPPORTED_HORIZONS}")
    frame = _as_dataframe(data)
    if log_return_column not in frame:
        frame = _ensure_log_returns(frame)
        log_return_column = "log_return"
    squared = frame[log_return_column].pow(2)
    future_mean = (
        squared.shift(-1)
        .iloc[::-1]
        .rolling(horizon, min_periods=horizon)
        .mean()
        .iloc[::-1]
    )
    scale = np.sqrt(252.0 / horizon) if annualize else 1.0
    target = pd.Series(
        np.sqrt(future_mean) * scale, index=frame.index, dtype=np.float64
    )
    target.name = name or f"future_realized_volatility_{horizon}d"
    return target


def build_supervised_dataset(
    data: pd.DataFrame | OHLCVData,
    *,
    horizon: int,
    feature_config: FeatureConfig | Mapping[str, Any] | None = None,
    **legacy_options: object,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build aligned and validated X/y, dropping rows only after joining."""
    if legacy_options:
        feature_config = _legacy_config(legacy_options)
    source = _as_dataframe(data)
    features, _ = build_feature_matrix(source, config=feature_config)
    target = make_volatility_target(source, horizon=horizon)
    aligned = features.join(target, how="inner")
    aligned = aligned.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if aligned.empty:
        raise ValueError("Supervised dataset is empty after alignment")
    y = aligned.pop(str(target.name))
    X = aligned
    validate_supervised_dataset(X, y, input_rows=len(source))
    return X, y


def validate_supervised_dataset(
    X: pd.DataFrame, y: pd.Series, *, input_rows: int | None = None
) -> DatasetValidationReport:
    """Validate model-readiness and return row-removal statistics."""
    if not isinstance(X.index, pd.DatetimeIndex) or not isinstance(
        y.index, pd.DatetimeIndex
    ):
        raise ValueError("X and y must use DatetimeIndex")
    if X.index.has_duplicates or y.index.has_duplicates:
        raise ValueError("X and y indices must not contain duplicates")
    if not X.index.equals(y.index):
        raise ValueError("X and y indices are not aligned")
    if X.columns.has_duplicates:
        raise ValueError("feature names must be unique")
    if X.isna().any().any() or y.isna().any():
        raise ValueError("X and y must not contain NaN values")
    if (
        not np.isfinite(X.to_numpy(dtype=np.float64)).all()
        or not np.isfinite(y.to_numpy(dtype=np.float64)).all()
    ):
        raise ValueError("X and y must contain only finite values")
    constant = [column for column in X if X[column].nunique(dropna=False) <= 1]
    if constant:
        raise ValueError(f"constant features are not allowed: {constant}")
    original = len(X) if input_rows is None else input_rows
    return DatasetValidationReport(original, len(X), original - len(X), X.shape[1])


def build_volatility_features(data: pd.DataFrame | OHLCVData) -> pd.DataFrame:
    """Build and clean the default configured feature matrix."""
    features, _ = build_feature_matrix(data)
    return features.replace([np.inf, -np.inf], np.nan).dropna()


def add_lagged_returns(
    data: pd.DataFrame | OHLCVData, *, lags: tuple[int, ...] = (1, 5)
) -> pd.DataFrame:
    """Compatibility helper adding shifted simple adjusted-price returns."""
    frame = add_simple_returns(data)
    for lag in lags:
        frame[f"return_lag_{lag}"] = frame["simple_return"].shift(lag)
    return frame


def add_lagged_volatility(
    data: pd.DataFrame | OHLCVData,
    *,
    volatility_column: str = "historical_volatility_5",
    lags: tuple[int, ...] = (1,),
) -> pd.DataFrame:
    """Compatibility helper adding shifted historical volatility."""
    frame = _as_dataframe(data)
    if volatility_column not in frame:
        frame = add_realized_volatility(frame, windows=(5,))
    for lag in lags:
        frame[f"volatility_lag_{lag}"] = frame[volatility_column].shift(lag)
    return frame


def _wilder_average(values: pd.Series, period: int) -> pd.Series:
    """Return Wilder smoothing seeded with the first period arithmetic mean."""
    result = pd.Series(np.nan, index=values.index, dtype=np.float64)
    valid_positions = np.flatnonzero(values.notna().to_numpy())
    if len(valid_positions) < period:
        return result
    seed_position = int(valid_positions[period - 1])
    seed = values.iloc[valid_positions[:period]].mean()
    result.iloc[seed_position] = np.float64(seed)
    for position in range(seed_position + 1, len(values)):
        result.iloc[position] = np.float64(
            (result.iloc[position - 1] * (period - 1) + values.iloc[position]) / period
        )
    return result


def _as_dataframe(data: pd.DataFrame | OHLCVData) -> pd.DataFrame:
    if isinstance(data, OHLCVData):
        return data.to_dataframe()
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("feature data index must be a DatetimeIndex")
    return data.copy(deep=True)


def _ensure_log_returns(frame: pd.DataFrame) -> pd.DataFrame:
    if "log_return" not in frame:
        return add_log_returns(frame)
    return frame


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"feature data is missing required columns: {missing}")


def _validate_windows(label: str, windows: tuple[int, ...]) -> None:
    if any(window < 2 for window in windows):
        raise ValueError(f"{label} windows must be at least 2")
    if len(set(windows)) != len(windows):
        raise ValueError(f"{label} windows must be unique")


def _resolve_config(config: FeatureConfig | Mapping[str, Any] | None) -> FeatureConfig:
    if config is None:
        return FeatureConfig()
    if isinstance(config, FeatureConfig):
        return config
    return FeatureConfig.from_mapping(config)


def _bool_value(section: Mapping[str, Any], key: str, default: bool) -> bool:
    value = section.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be boolean")
    return value


def _int_value(section: Mapping[str, Any], key: str, default: int) -> int:
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _windows_value(
    section: Mapping[str, Any], key: str, default: tuple[int, ...]
) -> tuple[int, ...]:
    value = section.get(key, default)
    if not isinstance(value, (list, tuple)) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise ValueError(f"{key} must be a list of integers")
    return tuple(value)


def _legacy_config(options: Mapping[str, object]) -> FeatureConfig:
    allowed = {
        "volatility_windows",
        "return_lags",
        "volatility_lags",
        "moving_average_windows",
    }
    unknown = set(options) - allowed
    if unknown:
        raise TypeError(f"unexpected feature options: {sorted(unknown)}")
    return FeatureConfig(
        volatility_windows=tuple(options.get("volatility_windows", (5, 21))),  # type: ignore[arg-type]
        sma_windows=tuple(options.get("moving_average_windows", (5, 21))),  # type: ignore[arg-type]
        ema_windows=(),
        rsi_enabled=False,
        atr_enabled=False,
        volume_enabled=True,
    )


__all__ = [
    "DatasetValidationReport",
    "FeatureConfig",
    "FeatureMetadata",
    "SUPPORTED_HORIZONS",
    "add_atr",
    "add_exponential_moving_averages",
    "add_lagged_returns",
    "add_lagged_volatility",
    "add_log_returns",
    "add_moving_averages",
    "add_realized_volatility",
    "add_rsi",
    "add_simple_returns",
    "add_volume_change",
    "add_volume_features",
    "build_feature_matrix",
    "build_supervised_dataset",
    "build_volatility_features",
    "make_volatility_target",
    "validate_supervised_dataset",
]
