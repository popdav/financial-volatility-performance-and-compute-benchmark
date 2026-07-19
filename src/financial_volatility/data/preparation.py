"""Validated, reproducible research-dataset preparation."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from financial_volatility.config import DatasetSettings, load_dataset_settings
from financial_volatility.data.cache import (
    cache_path,
    load_ohlcv_cache,
    save_ohlcv_cache,
)
from financial_volatility.data.types import MarketDataValidationError, OHLCVData
from financial_volatility.data.yahoo import YahooDownloader, fetch_yahoo_frame

RESEARCH_COLUMNS = ("open", "high", "low", "close", "adjusted_close", "volume")


@dataclass(frozen=True)
class PreparationResult:
    """Paths and quality counts produced by dataset preparation."""

    dataset_path: Path
    metadata_path: Path
    report_path: Path
    summary_path: Path
    cache_path: Path
    duplicate_rows_removed: int
    invalid_rows_removed: int


def prepare_dataset(
    config_path: str | Path,
    *,
    force_refresh: bool = False,
    downloader: YahooDownloader | None = None,
) -> PreparationResult:
    """Acquire, validate, cache, export, and inspect a configured dataset."""
    settings = load_dataset_settings(config_path)
    if settings.provider != "yahoo_finance":
        raise ValueError(f"Unsupported research dataset provider: {settings.provider}")
    if not settings.symbol or not settings.start_date or not settings.end_date:
        raise ValueError("dataset symbol, start_date, and end_date are required")

    cached_path = cache_path(
        settings.cache_directory,
        provider=settings.provider,
        symbol=settings.symbol,
        start=settings.start_date,
        end=settings.end_date,
        frequency=settings.frequency,
    )
    data = (
        None
        if force_refresh
        else load_ohlcv_cache(
            cached_path, symbol=settings.symbol, provider=settings.provider
        )
    )
    duplicate_count = 0
    invalid_count = 0
    original_missing: dict[str, int]
    if data is None:
        raw = _download_raw(settings, downloader)
        normalized, duplicate_count, invalid_count, original_missing = (
            _clean_and_validate(raw, settings)
        )
        data = OHLCVData(normalized, symbol=settings.symbol, provider=settings.provider)
        save_ohlcv_cache(data, cached_path)
    else:
        frame = data.to_dataframe()
        original_missing = {
            str(column): int(frame[column].isna().sum()) for column in frame
        }
        _validate_frame(frame, settings)

    frame = data.to_dataframe()
    start_year, end_year = settings.start_date.year, settings.end_date.year
    stem = f"{settings.symbol.lower()}_{settings.frequency}_{start_year}_{end_year}"
    settings.output_directory.mkdir(parents=True, exist_ok=True)
    dataset_path = settings.output_directory / f"{stem}.parquet"
    metadata_path = settings.output_directory / f"{stem}_metadata.json"
    frame.to_parquet(dataset_path)

    metadata = _metadata(
        settings, frame, cached_path, original_missing, duplicate_count, invalid_count
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    report_path, summary_path = _write_inspection_outputs(settings, frame, metadata)
    return PreparationResult(
        dataset_path,
        metadata_path,
        report_path,
        summary_path,
        cached_path,
        duplicate_count,
        invalid_count,
    )


def _download_raw(
    settings: DatasetSettings, downloader: YahooDownloader | None
) -> pd.DataFrame:
    """Download through the public loader and return its normalized frame."""
    return fetch_yahoo_frame(
        settings.symbol or "",
        start=settings.start_date,
        end=settings.end_date,
        frequency=settings.frequency,
        downloader=downloader,
    )


def _clean_and_validate(
    frame: pd.DataFrame, settings: DatasetSettings
) -> tuple[pd.DataFrame, int, int, dict[str, int]]:
    """Apply the documented conservative row policy and validate the result."""
    result = frame.copy(deep=True).sort_index()
    duplicate_count = int(result.attrs.get("duplicate_rows_removed", 0))
    duplicate_count += int(result.index.duplicated(keep="last").sum())
    result = result.loc[~result.index.duplicated(keep="last")]
    missing = {column: int(result[column].isna().sum()) for column in RESEARCH_COLUMNS}
    ohlc = ["open", "high", "low", "close"]
    if result.loc[:, ohlc].isna().all(axis=1).any():
        raise MarketDataValidationError("Rows with all OHLC prices missing are invalid")
    invalid = _invalid_rows(result)
    invalid_count = int(invalid.sum())
    if invalid_count and not settings.drop_invalid_rows:
        raise MarketDataValidationError(
            f"Dataset contains {invalid_count} invalid or partially missing rows; "
            "set dataset.drop_invalid_rows=true to remove them explicitly"
        )
    if invalid_count:
        result = result.loc[~invalid]
    _validate_frame(result, settings)
    return result, duplicate_count, invalid_count, missing


def _invalid_rows(frame: pd.DataFrame) -> pd.Series:
    prices = frame.loc[:, RESEARCH_COLUMNS[:-1]].apply(pd.to_numeric, errors="coerce")
    volume = pd.to_numeric(frame["volume"], errors="coerce")
    invalid = (
        prices.isna().any(axis=1)
        | volume.isna()
        | (prices <= 0).any(axis=1)
        | (frame["high"] < frame["low"])
        | (volume < 0)
        | ~np.isfinite(prices).all(axis=1)
        | ~np.isfinite(volume)
    )
    return cast(pd.Series, invalid)


def _validate_frame(frame: pd.DataFrame, settings: DatasetSettings) -> None:
    if frame.empty:
        raise MarketDataValidationError("Research dataset must be non-empty")
    missing = [column for column in RESEARCH_COLUMNS if column not in frame.columns]
    if missing:
        raise MarketDataValidationError(f"Research dataset missing columns: {missing}")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise MarketDataValidationError(
            "Research dataset index must be a DatetimeIndex"
        )
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise MarketDataValidationError(
            "Research dataset index must be unique and increasing"
        )
    if _invalid_rows(frame).any():
        raise MarketDataValidationError(
            "Research dataset contains invalid market values"
        )
    if settings.start_date is None or settings.end_date is None:
        raise MarketDataValidationError("Requested start and end dates are required")
    requested_start = pd.Timestamp(settings.start_date)
    requested_end = pd.Timestamp(settings.end_date)
    if frame.index[0].normalize() < requested_start:
        raise MarketDataValidationError(
            "First observation is earlier than requested start"
        )
    if frame.index[-1].normalize() > requested_end:
        raise MarketDataValidationError("Last observation is later than requested end")
    if (requested_end - frame.index[-1].normalize()).days > 10:
        raise MarketDataValidationError(
            "Last observation is implausibly far before requested end"
        )


def _metadata(
    settings: DatasetSettings,
    frame: pd.DataFrame,
    cached_path: Path,
    missing: dict[str, int],
    duplicates: int,
    invalid: int,
) -> dict[str, object]:
    return {
        "provider": settings.provider,
        "symbol": settings.symbol,
        "requested_date_range": [str(settings.start_date), str(settings.end_date)],
        "actual_date_range": [str(frame.index[0].date()), str(frame.index[-1].date())],
        "frequency": settings.frequency,
        "row_count": len(frame),
        "columns": list(frame.columns),
        "missing_value_counts": missing,
        "duplicate_rows_removed": duplicates,
        "invalid_rows_removed": invalid,
        "cache_path": str(cached_path),
        "creation_timestamp": datetime.now(UTC).isoformat(),
        "git_commit_hash": _git_commit(),
        "package_versions": {
            name: _version(name) for name in ("pandas", "pyarrow", "yfinance")
        },
    }


def _write_inspection_outputs(
    settings: DatasetSettings, frame: pd.DataFrame, metadata: dict[str, object]
) -> tuple[Path, Path]:
    output = Path("results/dataset")
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    adjusted = frame[settings.return_price_column]
    returns = np.log(adjusted).diff()
    volatility = returns.rolling(21).std() * np.sqrt(252)
    for values, title, filename in (
        (adjusted, "SPY adjusted price", "adjusted_price.png"),
        (returns, "SPY daily log returns", "daily_log_returns.png"),
        (
            volatility,
            "SPY rolling 21-day historical volatility",
            "rolling_21d_volatility.png",
        ),
    ):
        figure, axis = plt.subplots(figsize=(10, 4))
        values.plot(ax=axis, title=title)
        figure.tight_layout()
        figure.savefig(plots / filename, dpi=150)
        plt.close(figure)
    index = cast(pd.DatetimeIndex, frame.index)
    yearly = frame.groupby(index.year).size()
    valid_returns = returns.dropna()
    report = _report_markdown(frame, metadata, yearly, valid_returns)
    report_path = output / "spy_dataset_report.md"
    report_path.write_text(report, encoding="utf-8")
    summary = pd.DataFrame({"year": yearly.index, "observation_count": yearly.values})
    summary_path = output / "spy_dataset_summary.csv"
    summary.to_csv(summary_path, index=False)
    return report_path, summary_path


def _report_markdown(
    frame: pd.DataFrame,
    metadata: dict[str, object],
    yearly: pd.Series,
    returns: pd.Series,
) -> str:
    statistics = cast(pd.DataFrame, frame.loc[:, RESEARCH_COLUMNS].describe())
    stats = _text_table(statistics)
    first_last = _text_table(pd.concat([frame.head(1), frame.tail(1)]))
    missing = _text_table(
        pd.Series(metadata["missing_value_counts"], name="missing").to_frame()
    )
    counts = _text_table(yearly.rename("observations").to_frame())
    maximum = pd.Timestamp(returns.idxmax())
    minimum = pd.Timestamp(returns.idxmin())
    positive = f"{returns.loc[maximum]:.8f} ({maximum.date()})"
    negative = f"{returns.loc[minimum]:.8f} ({minimum.date()})"
    return f"""# SPY Dataset Inspection Report

- Requested range: {metadata["requested_date_range"]}
- Actual range: {metadata["actual_date_range"]}
- Observations: {metadata["row_count"]}
- Largest positive daily adjusted-price return: {positive}
- Largest negative daily adjusted-price return: {negative}

The configured range includes the dot-com period, global financial crisis, and
COVID-19 period. This is calendar coverage only, not algorithmic regime labeling.

## First and last rows

{first_last}

## Missing values

{missing}

## Descriptive statistics

{stats}

## Observations by year

{counts}
"""


def _text_table(frame: pd.DataFrame) -> str:
    """Render a dependency-free fixed-width table for Markdown reports."""
    return f"```text\n{frame.to_string()}\n```"


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


__all__ = ["PreparationResult", "prepare_dataset"]
