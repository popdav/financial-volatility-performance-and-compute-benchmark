# Dataset Research Plan

The thesis uses daily SPY OHLCV data from Yahoo Finance from 1999-01-01 through
the latest fully completed calendar year (currently configured as 2025-12-31).
This range includes the dot-com period, the global financial crisis, and the
COVID-19 period as calendar coverage; those labels are not inferred regimes.

The preparation stage acquires data, normalizes raw and adjusted prices,
deduplicates dates, applies conservative quality checks, caches the response,
and produces descriptive inspection artifacts. It never fills absent trading
days and never interpolates prices. Partially invalid rows cause failure unless
`drop_invalid_rows: true` is explicitly configured, in which case removal counts
are recorded.

Outputs are:

- `data/processed/spy_daily_1999_2025.parquet`
- `data/processed/spy_daily_1999_2025_metadata.json`
- `results/dataset/spy_dataset_report.md`
- `results/dataset/spy_dataset_summary.csv`
- inspection PNGs under `results/dataset/plots/`

Target construction and model feature engineering remain separate future steps.
