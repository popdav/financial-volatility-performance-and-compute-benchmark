"""Yahoo Finance OHLCV downloader."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import date
from typing import Any, cast

import pandas as pd

from financial_volatility.data.types import MarketDataValidationError, OHLCVData

logger = logging.getLogger(__name__)

YahooDownloader = Callable[..., pd.DataFrame]


def download_yahoo_ohlcv(
    symbol: str,
    *,
    start: str | date | pd.Timestamp | None = None,
    end: str | date | pd.Timestamp | None = None,
    frequency: str = "daily",
    downloader: YahooDownloader | None = None,
) -> OHLCVData:
    """Download OHLCV market data from Yahoo Finance."""
    frame = fetch_yahoo_frame(
        symbol, start=start, end=end, frequency=frequency, downloader=downloader
    )
    return OHLCVData(frame, symbol=symbol, provider="yahoo_finance")


def fetch_yahoo_frame(
    symbol: str,
    *,
    start: str | date | pd.Timestamp | None = None,
    end: str | date | pd.Timestamp | None = None,
    frequency: str = "daily",
    downloader: YahooDownloader | None = None,
) -> pd.DataFrame:
    """Fetch and normalize Yahoo data without applying the OHLCV value contract."""
    if not symbol.strip():
        raise ValueError("symbol must be non-empty")

    logger.info("Downloading Yahoo Finance data for %s", symbol)
    download = downloader or _default_yfinance_downloader
    if frequency != "daily":
        raise ValueError(f"Unsupported Yahoo Finance frequency: {frequency}")
    # yfinance defines `end` as exclusive. The public loader contract is inclusive,
    # so request the next calendar day and validate the configured boundary later.
    provider_end = None if end is None else pd.Timestamp(end) + pd.Timedelta(days=1)
    raw_data = download(
        symbol,
        start=start,
        end=provider_end,
        interval="1d",
        auto_adjust=False,
        actions=False,
        progress=False,
    )
    if not isinstance(raw_data, pd.DataFrame) or raw_data.empty:
        raise ValueError(f"Yahoo Finance returned no data for symbol: {symbol}")

    return _normalize_yahoo_frame(raw_data)


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
        frame = frame.rename(columns={"adj_close": "adjusted_close"})
    elif "adjusted_close" not in frame.columns:
        raise MarketDataValidationError(
            "Yahoo Finance response has no separate adjusted-close column. "
            "The loader requests auto_adjust=False, so adjusted-price semantics "
            "cannot be determined safely."
        )

    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index, errors="raise")

    index = pd.DatetimeIndex(frame.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    frame.index = index
    columns = ["open", "high", "low", "close", "adjusted_close", "volume"]
    frame = frame.loc[:, columns].sort_index()
    duplicate_count = int(frame.index.duplicated(keep="last").sum())
    frame = frame.loc[~frame.index.duplicated(keep="last")]
    frame.attrs["duplicate_rows_removed"] = duplicate_count
    return frame


__all__ = ["YahooDownloader", "download_yahoo_ohlcv", "fetch_yahoo_frame"]
