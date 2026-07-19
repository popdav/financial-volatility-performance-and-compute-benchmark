# Repository Audit Report

Date: 2026-07-10

## Summary

The repository has been audited against `docs/architecture.md`,
`docs/roadmap.md`, and `docs/backlog.md`. The core benchmark platform is
implemented and verified end-to-end for local synthetic and local CSV
experiments using real registered model adapters.

Final status: ready for small controlled thesis pilot experiments. Run one
pilot per model and inspect outputs before launching full-scale thesis runs.

## Fixes Applied During Audit

- Added missing `pytest-cov` dev dependency so the required coverage command
  works.
- Added direct runtime dependencies for `yfinance` and `joblib`.
- Rewired `ExperimentPipeline` to use future realized-volatility targets from
  `build_supervised_dataset` instead of using rolling volatility features as
  targets.
- Separated configured forecast target horizon from test-window prediction
  count in `BenchmarkRunner`.
- Added sequence conversion in the experiment pipeline for LSTM and Transformer
  adapters.
- Removed the CLI dummy model from public `list-models`; CLI smoke tests now
  use registered real models.
- Improved CLI errors for missing or invalid config files.
- Tightened OHLCV validation to reject missing, non-numeric, and non-finite
  required values.
- Added the missing `volume_change` feature required by the backlog.
- Added target coverage for horizon 21.
- Removed generated `__pycache__` artifacts.
- Updated README, roadmap, and backlog statuses.

## Implementation Status

### Project Quality

Implemented and verified:

- Standard `src/financial_volatility` layout.
- Ruff, pytest, mypy, and pre-commit configuration.
- Canonical package namespace is `financial_volatility`.
- Model construction uses `ModelRegistry`; no large model-selection if/elif
  chain exists for registered real models.
- Generated Python cache artifacts were removed.

Dependencies are aligned with direct imports. No unused runtime dependency was
identified in the implemented modules.

### Data Layer

Implemented and verified:

- Local CSV loading with configurable date column, lowercase columns, datetime
  index conversion, and chronological sorting.
- Yahoo Finance loading with mocked test coverage and declared `yfinance`
  dependency.
- Parquet cache read/write and deterministic cache paths.
- OHLCV validation for `DatetimeIndex`, chronological ordering, duplicate
  timestamps, required columns, empty frames, missing values, non-numeric
  values, and non-finite values.

Tests mock Yahoo network access; default test runs do not require network.

### Feature Engineering

Implemented and verified:

- Simple returns and log returns.
- Rolling realized volatility for 5-day and 21-day windows.
- Return and volatility lag features.
- Moving averages.
- Volume change.
- Future realized-volatility targets for horizons 1, 5, and 21.
- Aligned `X` and `y` with dropped warm-up and unavailable-future rows.

No lookahead was found in target construction: target volatility is built from
returns strictly after the feature timestamp.

### Splitting And Validation

Implemented and verified:

- Chronological train/test split by float test size and split date.
- Expanding and rolling walk-forward splitter.
- Boundary tests for empty splits, invalid dates, fold boundaries, and
  no-overlap behavior.

### Models

Implemented and verified:

- `GARCHModel`
- `LinearRegressionModel`
- `XGBoostModel`
- `LSTMModel`
- `TransformerModel`

Each implements `ForecastModel`, supports train/predict, metadata, CPU
execution, and save/load tests. CUDA handling is optional and skipped or fails
clearly when unavailable. Tenstorrent support is optional and does not affect
normal environments.

Deviation: the Transformer/PatchTST backlog item is implemented as a compact
PyTorch Transformer encoder, not a true PatchTST architecture.

### Benchmarking

Implemented and verified:

- RMSE, MAE, MAPE.
- Training and inference timing.
- Peak Python memory measurement.
- Model size estimation.
- Stable CSV metric columns.
- Metrics are calculated against aligned prediction and target arrays.

Throughput is not implemented.

### Experiment Pipeline

Verified end-to-end:

- Synthetic CLI run using `configs/default.yaml`.
- Temporary local CSV CLI run using a real generated CSV file.
- Data loading, feature engineering, target construction, splitting, model
  creation, training, prediction, benchmarking, and result persistence.

Failure messages are now concise for missing or invalid config files.

### Configuration And Registry

Implemented and verified:

- YAML config validation with Pydantic.
- Unknown model names fail clearly through `ModelRegistry`.
- Non-scalar model parameters fail clearly in config-driven construction.
- Default config is a runnable synthetic experiment.

### CLI

Verified:

- `fvbench --help`
- `fvbench list-models`
- `fvbench validate-config --config configs/default.yaml`
- `fvbench run --config configs/default.yaml`
- Invalid config path exits non-zero with a concise error.

### Results And Visualization

Implemented and verified:

- CSV writing and append without duplicate headers.
- Automatic output directory creation.
- Plot generation from result CSV files.
- Deterministic thesis table outputs.

Remaining limitation: result storage persists aggregate metric rows, not
per-timestamp prediction files.

### Tests

Current coverage summary:

- 159 tests passed.
- Coverage: 90% total for `financial_volatility`.

Coverage is strongest around data, features, splitting, metrics, storage, and
configuration. Lower-coverage areas are mostly optional/default error paths and
model validation branches.

## Exact Commands Used

```bash
uv sync
uv add --dev pytest-cov
uv add yfinance joblib
uv run pytest
uv run pytest --cov=financial_volatility --cov-report=term-missing
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run fvbench --help
uv run fvbench list-models
uv run fvbench validate-config --config configs/default.yaml
uv run fvbench run --config configs/default.yaml
uv run fvbench validate-config --config /tmp/fvbench-local-smoke/config.yaml
uv run fvbench run --config /tmp/fvbench-local-smoke/config.yaml
uv run fvbench run --config /tmp/fvbench-local-smoke/missing.yaml
uv run pre-commit run --all-files
```

## Remaining Risks

- Yahoo Finance live downloads were not exercised against the network in this
  audit; tests use mocked downloader responses.
- The Transformer implementation is not PatchTST.
- The default result CSV does not store per-timestamp predictions.
- Peak memory measurement uses Python `tracemalloc`; it does not capture all
  native library or GPU memory.
- Full thesis readiness still requires pilot runs on the actual thesis datasets
  and manual review of model outputs.

## Readiness Statement

The repository is ready for real thesis pilot experiments on local datasets.
It is not yet proven ready for final thesis-scale experiments until each target
dataset and each selected model has completed a pilot run with reviewed metrics,
runtime, and output files.
