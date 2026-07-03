"""Yahoo Finance downloader tests."""

import pandas as pd
import pytest

from financial_volatility.data.yahoo import download_yahoo_ohlcv


def test_yahoo_downloader_converts_mocked_response_to_ohlcv_data() -> None:
    """A mocked Yahoo response is normalized to OHLCVData."""
    data = download_yahoo_ohlcv(
        "SPY",
        start="2026-01-01",
        end="2026-01-05",
        downloader=_mock_downloader,
    )

    frame = data.to_dataframe()
    assert data.symbol == "SPY"
    assert data.provider == "yahoo_finance"
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
    assert frame["close"].iloc[0] == pytest.approx(100.5)


def test_yahoo_downloader_rejects_empty_response() -> None:
    """Empty Yahoo responses fail clearly."""
    with pytest.raises(ValueError, match="no data"):
        download_yahoo_ohlcv(
            "MISSING",
            downloader=lambda *_args, **_kwargs: pd.DataFrame(),
        )


def test_yahoo_downloader_rejects_invalid_symbol() -> None:
    """Symbols must be non-empty."""
    with pytest.raises(ValueError, match="symbol"):
        download_yahoo_ohlcv(" ", downloader=_mock_downloader)


def _mock_downloader(*_args, **_kwargs) -> pd.DataFrame:
    """Return deterministic Yahoo-shaped data."""
    index = pd.date_range("2026-01-01", periods=3, freq="D")
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0],
            "High": [101.0, 102.0, 103.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [100.0, 101.0, 102.0],
            "Adj Close": [100.5, 101.5, 102.5],
            "Volume": [1000, 1100, 1200],
        },
        index=index,
    )
