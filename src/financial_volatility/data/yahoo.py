"""Yahoo Finance OHLCV downloader."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import date
from typing import Any, cast

import pandas as pd

from financial_volatility.data.types import OHLCVData

logger = logging.getLogger(__name__)

YahooDownloader = Callable[..., pd.DataFrame]


def download_yahoo_ohlcv(
    symbol: str,
    *,
    start: str | date | pd.Timestamp | None = None,
    end: str | date | pd.Timestamp | None = None,
    downloader: YahooDownloader | None = None,
) -> OHLCVData:
    """Download OHLCV market data from Yahoo Finance."""
    if not symbol.strip():
        raise ValueError("symbol must be non-empty")

    logger.info("Downloading Yahoo Finance data for %s", symbol)
    download = downloader or _default_yfinance_downloader
    raw_data = download(symbol, start=start, end=end, progress=False)
    if not isinstance(raw_data, pd.DataFrame) or raw_data.empty:
        raise ValueError(f"Yahoo Finance returned no data for symbol: {symbol}")

    frame = _normalize_yahoo_frame(raw_data)
    return OHLCVData(frame, symbol=symbol, provider="yahoo_finance")


def _default_yfinance_downloader(*args: Any, **kwargs: Any) -> pd.DataFrame:
    """Call yfinance.download when yfinance is installed."""
    try:
        import yfinance as yf  # type: ignore[import-untyped]
    except ImportError as error:
        msg = "yfinance is required for default Yahoo Finance downloads"
        raise RuntimeError(msg) from error

    return cast(pd.DataFrame, yf.download(*args, **kwargs))


def _normalize_yahoo_frame(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Normalize Yahoo Finance columns to the OHLCV contract."""
    frame = raw_data.copy(deep=True)
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)

    frame.columns = [
        str(column).strip().lower().replace(" ", "_") for column in frame.columns
    ]
    if "adj_close" in frame.columns:
        frame["close"] = frame["adj_close"]

    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index, errors="raise")

    return frame.loc[:, ["open", "high", "low", "close", "volume"]].sort_index()


__all__ = ["YahooDownloader", "download_yahoo_ohlcv"]
