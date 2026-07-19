"""Parquet cache helpers for reproducible OHLCV data."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from financial_volatility.data.types import OHLCVData


def cache_path(
    cache_dir: str | Path,
    *,
    provider: str,
    symbol: str,
    start: str | date | pd.Timestamp | None,
    end: str | date | pd.Timestamp | None,
    frequency: str = "daily",
) -> Path:
    """Build a deterministic cache path for one market data request."""
    filename = "_".join(
        [
            _slug(provider),
            _slug(symbol),
            _date_part(start),
            _date_part(end),
            _slug(frequency),
        ]
    )
    return Path(cache_dir) / f"{filename}.parquet"


def save_ohlcv_cache(data: OHLCVData, path: str | Path) -> Path:
    """Save OHLCV data to a parquet cache file."""
    cache_file = Path(path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    data.to_dataframe().to_parquet(cache_file)
    return cache_file


def load_ohlcv_cache(
    path: str | Path,
    *,
    symbol: str | None = None,
    provider: str | None = None,
) -> OHLCVData | None:
    """Load OHLCV data from parquet, returning None when the cache is missing."""
    cache_file = Path(path)
    if not cache_file.exists():
        return None

    frame = pd.read_parquet(cache_file)
    return OHLCVData(frame, symbol=symbol, provider=provider)


def _slug(value: str) -> str:
    """Normalize a cache key component."""
    return value.strip().lower().replace("/", "-").replace(" ", "-")


def _date_part(value: str | date | pd.Timestamp | None) -> str:
    """Normalize optional dates for cache filenames."""
    if value is None:
        return "none"

    return pd.Timestamp(value).strftime("%Y%m%d")


__all__ = ["cache_path", "load_ohlcv_cache", "save_ohlcv_cache"]
