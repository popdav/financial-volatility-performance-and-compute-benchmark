"""Local market data loaders."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from financial_volatility.data.types import OHLCVData


def load_ohlcv_csv(
    path: str | Path,
    *,
    date_column: str = "date",
    symbol: str | None = None,
    provider: str | None = "csv",
) -> OHLCVData:
    """Load OHLCV market data from a local CSV file."""
    csv_path = Path(path)
    frame = pd.read_csv(csv_path)
    frame.columns = [str(column).lower() for column in frame.columns]
    normalized_date_column = date_column.lower()

    if normalized_date_column not in frame.columns:
        msg = f"CSV file {csv_path} is missing date column: {date_column}"
        raise ValueError(msg)

    try:
        frame[normalized_date_column] = pd.to_datetime(
            frame[normalized_date_column],
            errors="raise",
        )
    except (TypeError, ValueError) as error:
        msg = f"CSV file {csv_path} contains invalid dates in column: {date_column}"
        raise ValueError(msg) from error

    frame = frame.set_index(normalized_date_column).sort_index()

    return OHLCVData(data=frame, symbol=symbol, provider=provider)


__all__ = ["load_ohlcv_csv"]
