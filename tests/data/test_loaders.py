"""Local CSV market data loader tests."""

from pathlib import Path

import pandas as pd
import pytest

from financial_volatility.data.loaders import load_ohlcv_csv
from financial_volatility.data.types import OHLCVData


def test_valid_csv_loads_successfully(tmp_path: Path) -> None:
    """A valid local CSV is loaded into OHLCVData."""
    csv_path = tmp_path / "prices.csv"
    csv_path.write_text(
        """Date,Open,High,Low,Close,Volume
2026-01-02,2.0,3.0,1.5,2.5,200
2026-01-01,1.0,2.0,0.5,1.5,100
""",
        encoding="utf-8",
    )

    data = load_ohlcv_csv(csv_path, symbol="TEST")

    assert isinstance(data, OHLCVData)
    assert data.symbol == "TEST"
    assert data.provider == "csv"
    assert data.columns == ("open", "high", "low", "close", "volume")
    assert data.start_timestamp == pd.Timestamp("2026-01-01")
    assert data.end_timestamp == pd.Timestamp("2026-01-02")


def test_custom_date_column_works(tmp_path: Path) -> None:
    """The CSV loader supports configurable date column names."""
    csv_path = tmp_path / "prices.csv"
    csv_path.write_text(
        """TradeDate,Open,High,Low,Close,Volume
2026-01-01,1.0,2.0,0.5,1.5,100
""",
        encoding="utf-8",
    )

    data = load_ohlcv_csv(csv_path, date_column="TradeDate")

    assert data.start_timestamp == pd.Timestamp("2026-01-01")
    assert "tradedate" not in data.columns


def test_missing_date_column_raises_value_error(tmp_path: Path) -> None:
    """The configured date column must be present in the CSV."""
    csv_path = tmp_path / "prices.csv"
    csv_path.write_text(
        """timestamp,open,high,low,close,volume
2026-01-01,1.0,2.0,0.5,1.5,100
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing date column"):
        load_ohlcv_csv(csv_path)


def test_missing_ohlcv_columns_raise_value_error(tmp_path: Path) -> None:
    """Required OHLCV validation is delegated to OHLCVData."""
    csv_path = tmp_path / "prices.csv"
    csv_path.write_text(
        """date,open,high,low,close
2026-01-01,1.0,2.0,0.5,1.5
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required columns: volume"):
        load_ohlcv_csv(csv_path)


def test_invalid_dates_raise_value_error(tmp_path: Path) -> None:
    """Date parsing errors are reported as ValueError."""
    csv_path = tmp_path / "prices.csv"
    csv_path.write_text(
        """date,open,high,low,close,volume
not-a-date,1.0,2.0,0.5,1.5,100
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid dates"):
        load_ohlcv_csv(csv_path)
