"""Parquet cache tests."""

import pandas as pd

from financial_volatility.data.cache import (
    cache_path,
    load_ohlcv_cache,
    save_ohlcv_cache,
)
from financial_volatility.data.types import OHLCVData


def test_ohlcv_cache_roundtrip_works(tmp_path) -> None:
    """OHLCV parquet cache files can be written and read back."""
    data = OHLCVData(_ohlcv_frame(), symbol="SPY", provider="test")
    path = tmp_path / "nested" / "spy.parquet"

    saved_path = save_ohlcv_cache(data, path)
    loaded = load_ohlcv_cache(saved_path, symbol="SPY", provider="test")

    assert loaded is not None
    assert loaded.symbol == "SPY"
    assert loaded.provider == "test"
    pd.testing.assert_frame_equal(
        loaded.to_dataframe(),
        data.to_dataframe(),
        check_freq=False,
    )


def test_missing_cache_returns_none(tmp_path) -> None:
    """Missing cache files are handled cleanly."""
    assert load_ohlcv_cache(tmp_path / "missing.parquet") is None


def test_cache_path_is_deterministic(tmp_path) -> None:
    """Cache paths are deterministic for provider, symbol, and date range."""
    first = cache_path(
        tmp_path,
        provider="Yahoo Finance",
        symbol="SPY",
        start="2026-01-01",
        end="2026-02-01",
    )
    second = cache_path(
        tmp_path,
        provider="Yahoo Finance",
        symbol="SPY",
        start=pd.Timestamp("2026-01-01"),
        end=pd.Timestamp("2026-02-01"),
    )

    assert first == second
    assert first.name == "yahoo-finance_spy_20260101_20260201_daily.parquet"


def _ohlcv_frame() -> pd.DataFrame:
    """Create deterministic OHLCV data."""
    index = pd.date_range("2026-01-01", periods=3, freq="D")
    return pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [1000, 1100, 1200],
        },
        index=index,
    )
