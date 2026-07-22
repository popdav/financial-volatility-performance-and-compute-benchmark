"""Deterministic tests for leakage-safe historical features."""

import math

import numpy as np
import pandas as pd
import pytest

from financial_volatility.features.engineering import (
    FeatureConfig,
    add_atr,
    add_exponential_moving_averages,
    add_log_returns,
    add_moving_averages,
    add_realized_volatility,
    add_rsi,
    add_simple_returns,
    add_volume_features,
    build_feature_matrix,
)


def test_daily_returns_use_adjusted_close_and_preserve_index() -> None:
    frame = _frame()
    frame["close"] = frame["adjusted_close"] * 2
    simple = add_simple_returns(frame)
    logged = add_log_returns(frame)
    assert simple.index.equals(frame.index)
    assert pd.isna(logged["log_return"].iloc[0])
    expected = frame["adjusted_close"].iloc[1] / frame["adjusted_close"].iloc[0]
    assert logged["log_return"].iloc[1] == pytest.approx(math.log(expected))
    assert simple["simple_return"].iloc[1] == pytest.approx(expected - 1)


def test_returns_reject_missing_adjusted_close() -> None:
    with pytest.raises(ValueError, match="adjusted_close"):
        add_log_returns(_frame().drop(columns="adjusted_close"))


def test_rolling_volatility_is_sample_std_and_annualized() -> None:
    returns = add_log_returns(_frame())
    result = add_realized_volatility(returns, windows=(5,))
    expected = returns["log_return"].iloc[1:6].std() * np.sqrt(252)
    assert result["historical_volatility_5"].iloc[5] == pytest.approx(expected)
    assert result["historical_volatility_5"].iloc[:5].isna().all()


def test_sma_and_ema_windows_are_correct() -> None:
    frame = _frame()
    sma = add_moving_averages(frame, windows=(5,))
    ema = add_exponential_moving_averages(frame, windows=(5,))
    assert sma["sma_5"].iloc[4] == pytest.approx(
        frame["adjusted_close"].iloc[:5].mean()
    )
    expected_ema = (
        frame["adjusted_close"].ewm(span=5, adjust=False, min_periods=5).mean()
    )
    pd.testing.assert_series_equal(ema["ema_5"], expected_ema, check_names=False)


def test_rsi_uses_wilder_smoothing() -> None:
    frame = _frame(rows=20)
    result = add_rsi(frame, period=3)["rsi_3"]
    delta = frame["adjusted_close"].diff()
    gains = delta.clip(lower=0).iloc[1:4].mean()
    losses = (-delta.clip(upper=0)).iloc[1:4].mean()
    expected = 100 - 100 / (1 + gains / losses)
    assert result.iloc[3] == pytest.approx(expected)
    next_gain = max(delta.iloc[4], 0)
    next_loss = max(-delta.iloc[4], 0)
    expected_next = 100 - 100 / (
        1 + ((gains * 2 + next_gain) / 3) / ((losses * 2 + next_loss) / 3)
    )
    assert result.iloc[4] == pytest.approx(expected_next)


def test_atr_uses_true_range_and_wilder_smoothing() -> None:
    frame = _frame()
    frame.iloc[1, frame.columns.get_loc("high")] = 110
    result = add_atr(frame, period=3)["atr_3"]
    previous = frame["close"].shift(1)
    true_range = pd.concat(
        [
            (frame.high - frame.low),
            (frame.high - previous).abs(),
            (frame.low - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)
    assert result.iloc[2] == pytest.approx(true_range.iloc[:3].mean())
    assert result.iloc[3] == pytest.approx(
        (result.iloc[2] * 2 + true_range.iloc[3]) / 3
    )


def test_volume_features_are_log_and_percentage_change() -> None:
    frame = _frame()
    result = add_volume_features(frame)
    assert result["log_volume"].iloc[0] == pytest.approx(np.log(frame.volume.iloc[0]))
    assert result["volume_change"].iloc[1] == pytest.approx(
        frame.volume.iloc[1] / frame.volume.iloc[0] - 1
    )


def test_feature_config_controls_generation_and_metadata() -> None:
    config = FeatureConfig.from_mapping(
        {
            "features": {
                "price": {"enabled": False},
                "volatility": {"windows": [5]},
                "sma": {"windows": []},
                "ema": {"windows": []},
                "rsi": {"enabled": False},
                "atr": {"enabled": False},
                "volume": {"enabled": False},
            }
        }
    )
    features, metadata = build_feature_matrix(_frame(), config=config)
    assert list(features) == ["historical_volatility_5"]
    assert metadata[0].category == "volatility"
    assert metadata[0].rolling_window == 5


def test_features_at_t_do_not_change_when_future_data_changes() -> None:
    frame = _frame(rows=80)
    before, _ = build_feature_matrix(frame)
    changed = frame.copy()
    changed.iloc[61:, changed.columns.get_loc("adjusted_close")] *= 10
    changed.iloc[61:, changed.columns.get_loc("volume")] *= 7
    after, _ = build_feature_matrix(changed)
    pd.testing.assert_series_equal(before.iloc[60], after.iloc[60])


def _frame(rows: int = 90) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=rows, freq="D")
    adjusted = 100 + np.arange(rows) * 0.3 + np.sin(np.arange(rows))
    return pd.DataFrame(
        {
            "open": adjusted - 0.4,
            "high": adjusted + 1.2,
            "low": adjusted - 1,
            "close": adjusted + 0.1,
            "adjusted_close": adjusted,
            "volume": 1000 + np.arange(rows) ** 2,
        },
        index=index,
    )
