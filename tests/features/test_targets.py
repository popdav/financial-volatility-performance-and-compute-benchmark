"""Target alignment, cleaning, and validation tests."""

import numpy as np
import pandas as pd
import pytest

from financial_volatility.features.engineering import (
    FeatureConfig,
    build_supervised_dataset,
    make_volatility_target,
    validate_supervised_dataset,
)


@pytest.mark.parametrize("horizon", [5, 21])
def test_target_uses_exactly_next_h_returns_and_annualization(horizon: int) -> None:
    frame = _frame(100)
    returns = np.log(frame.adjusted_close / frame.adjusted_close.shift(1))
    target = make_volatility_target(frame, horizon=horizon)
    expected = np.sqrt(np.mean(np.square(returns.iloc[11 : 11 + horizon]))) * np.sqrt(
        252 / horizon
    )
    assert target.iloc[10] == pytest.approx(expected)
    assert target.iloc[-horizon:].isna().all()


@pytest.mark.parametrize("horizon", [0, 1, 2, 6, 22])
def test_target_rejects_unsupported_horizons(horizon: int) -> None:
    with pytest.raises(ValueError, match="horizon"):
        make_volatility_target(_frame(50), horizon=horizon)


def test_target_does_not_include_return_at_t() -> None:
    frame = _frame(50)
    frame["log_return"] = np.log(frame.adjusted_close / frame.adjusted_close.shift(1))
    original = make_volatility_target(frame, horizon=5)
    changed = frame.copy()
    changed.iloc[10, changed.columns.get_loc("log_return")] *= 100
    altered = make_volatility_target(changed, horizon=5)
    assert altered.iloc[10] == pytest.approx(original.iloc[10])


def test_supervised_builder_aligns_cleans_and_reports_removed_rows() -> None:
    frame = _frame(120)
    X, y = build_supervised_dataset(frame, horizon=21)
    report = validate_supervised_dataset(X, y, input_rows=len(frame))
    assert X.index.equals(y.index)
    assert X.index[0] == frame.index[63]
    assert X.index[-1] == frame.index[-22]
    assert not X.isna().any().any() and not y.isna().any()
    assert report.removed_rows == len(frame) - len(X)


def test_cleaning_occurs_after_alignment_and_handles_infinity() -> None:
    frame = _frame(90)
    frame.iloc[70, frame.columns.get_loc("volume")] = 0
    X, y = build_supervised_dataset(frame, horizon=5)
    assert np.isfinite(X.to_numpy()).all() and np.isfinite(y.to_numpy()).all()


def test_validation_rejects_constant_features() -> None:
    index = pd.date_range("2025-01-01", periods=3)
    with pytest.raises(ValueError, match="constant"):
        validate_supervised_dataset(
            pd.DataFrame({"x": [1, 1, 1]}, index=index),
            pd.Series([1, 2, 3], index=index),
        )


def test_builder_accepts_yaml_shaped_configuration() -> None:
    config = {
        "features": {
            "price": {"enabled": True},
            "volatility": {"windows": [5]},
            "sma": {"windows": [5]},
            "ema": {"windows": []},
            "rsi": {"enabled": False},
            "atr": {"enabled": False},
            "volume": {"enabled": True},
        }
    }
    X, _ = build_supervised_dataset(_frame(50), horizon=5, feature_config=config)
    assert list(X) == [
        "log_return",
        "simple_return",
        "historical_volatility_5",
        "sma_5",
        "log_volume",
        "volume_change",
    ]
    assert FeatureConfig.from_mapping(config).volatility_windows == (5,)


def _frame(rows: int) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=rows)
    adjusted = 100 + np.arange(rows) * 0.2 + np.sin(np.arange(rows) / 2)
    spread = 1 + 0.1 * np.sin(np.arange(rows) / 3)
    return pd.DataFrame(
        {
            "open": adjusted - 0.5,
            "high": adjusted + spread,
            "low": adjusted - spread,
            "close": adjusted,
            "adjusted_close": adjusted,
            "volume": 1000 + np.arange(rows) ** 2,
        },
        index=index,
    )
