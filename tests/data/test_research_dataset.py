"""Research dataset preparation and quality tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from financial_volatility.cli import app
from financial_volatility.data.preparation import _clean_and_validate, prepare_dataset
from financial_volatility.data.types import MarketDataValidationError
from financial_volatility.data.yahoo import download_yahoo_ohlcv


def test_normalization_sorts_deduplicates_and_removes_timezone() -> None:
    raw = _provider_frame().iloc[[2, 0, 1, 1]].copy()
    raw.index = raw.index.tz_localize("UTC")
    data = download_yahoo_ohlcv("SPY", downloader=lambda *_args, **_kwargs: raw)
    frame = data.to_dataframe()
    assert frame.index.is_monotonic_increasing
    assert frame.index.is_unique
    assert frame.index.tz is None
    assert list(frame.columns) == [
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
    ]


def test_missing_adjusted_price_semantics_fails() -> None:
    raw = _provider_frame().drop(columns="Adj Close")
    with pytest.raises(MarketDataValidationError, match="semantics"):
        download_yahoo_ohlcv("SPY", downloader=lambda *_args, **_kwargs: raw)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("High", 8.0, "invalid"),
        ("Open", -1.0, "invalid"),
        ("Volume", -1.0, "invalid"),
    ],
)
def test_invalid_market_values_fail(
    tmp_path: Path, column: str, value: float, message: str
) -> None:
    raw = _provider_frame()
    raw.loc[raw.index[0], column] = value
    settings = _settings(tmp_path)
    normalized = download_yahoo_ohlcv(
        "SPY", downloader=lambda *_args, **_kwargs: raw
    ).to_dataframe()
    with pytest.raises(MarketDataValidationError, match=message):
        _clean_and_validate(normalized, settings)


def test_explicit_invalid_row_removal_is_counted(tmp_path: Path) -> None:
    raw = _provider_frame()
    raw.loc[raw.index[0], "Volume"] = -1
    settings = _settings(tmp_path, drop_invalid_rows=True)
    normalized = download_yahoo_ohlcv(
        "SPY", downloader=lambda *_args, **_kwargs: raw
    ).to_dataframe()
    cleaned, duplicates, invalid, _missing = _clean_and_validate(normalized, settings)
    assert len(cleaned) == 2
    assert duplicates == 0
    assert invalid == 1


def test_prepare_cache_force_refresh_metadata_and_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config = _config(tmp_path)
    calls = 0

    def downloader(*_args: object, **_kwargs: object) -> pd.DataFrame:
        nonlocal calls
        calls += 1
        return _provider_frame()

    first = prepare_dataset(config, downloader=downloader)
    second = prepare_dataset(config, downloader=downloader)
    refreshed = prepare_dataset(config, force_refresh=True, downloader=downloader)
    assert calls == 2
    pd.testing.assert_frame_equal(
        pd.read_parquet(first.dataset_path), pd.read_parquet(second.dataset_path)
    )
    metadata = json.loads(refreshed.metadata_path.read_text(encoding="utf-8"))
    assert metadata["row_count"] == 3
    assert metadata["columns"][4] == "adjusted_close"
    assert first.report_path.exists()
    assert first.summary_path.exists()

    monkeypatch.setattr(
        "financial_volatility.data.preparation._default_yfinance_downloader",
        downloader,
        raising=False,
    )


def test_cli_dataset_prepare_with_mocked_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config = _config(tmp_path)
    monkeypatch.setattr(
        "financial_volatility.data.yahoo._default_yfinance_downloader",
        lambda *_args, **_kwargs: _provider_frame(),
    )
    result = CliRunner().invoke(app, ["dataset", "prepare", "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert "spy_daily_2025_2025.parquet" in result.output


def _provider_frame() -> pd.DataFrame:
    index = pd.to_datetime(["2025-01-07", "2025-01-08", "2025-01-09"])
    return pd.DataFrame(
        {
            "Open": [10.0, 11.0, 12.0],
            "High": [11.0, 12.0, 13.0],
            "Low": [9.0, 10.0, 11.0],
            "Close": [10.5, 11.5, 12.5],
            "Adj Close": [10.4, 11.4, 12.4],
            "Volume": [100, 110, 120],
        },
        index=index,
    )


def _config(tmp_path: Path) -> Path:
    path = tmp_path / "dataset.yaml"
    path.write_text(
        f"""dataset:
  provider: yahoo_finance
  symbol: SPY
  start_date: 2025-01-01
  end_date: 2025-01-10
  frequency: daily
  cache_directory: {tmp_path / "cache"}
  output_directory: {tmp_path / "processed"}
  return_price_column: adjusted_close
""",
        encoding="utf-8",
    )
    return path


def _settings(tmp_path: Path, *, drop_invalid_rows: bool = False):  # type: ignore[no-untyped-def]
    from financial_volatility.config import DatasetSettings

    return DatasetSettings(
        provider="yahoo_finance",
        symbol="SPY",
        start_date="2025-01-01",
        end_date="2025-01-10",
        cache_directory=tmp_path,
        output_directory=tmp_path,
        drop_invalid_rows=drop_invalid_rows,
    )
