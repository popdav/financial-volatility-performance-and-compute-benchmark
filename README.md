# Financial Volatility Performance and Compute Benchmark

Python research platform for benchmarking financial volatility forecasting
models across forecast accuracy and computational cost.

The project supports provider-agnostic OHLCV data structures, local CSV
experiments, volatility feature engineering, time-series splits, model adapters,
benchmark metrics, result storage, plots, and thesis-ready summary tables.

## Setup

Install dependencies with uv:

```bash
uv sync
```

Run quality checks:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest
uv run pytest --cov=financial_volatility --cov-report=term-missing
uv run pre-commit run --all-files
```

## Running Experiments

Validate a YAML config:

```bash
uv run fvbench validate-config --config configs/default.yaml
```

List available models:

```bash
uv run fvbench list-models
```

Run an experiment:

```bash
uv run fvbench run --config configs/default.yaml
```

`configs/default.yaml` is a minimal reproducible synthetic experiment. It
generates deterministic OHLCV data, builds future realized-volatility targets
with horizon 5, trains `linear_regression`, and writes `results/results.csv`.

The CLI supports registered model adapters: `garch`, `linear_regression`,
`xgboost`, `lstm`, and `transformer`. The config can pass model-specific scalar
parameters under `model.parameters`; sequence models require a `sequence_length`
parameter.

## Project Structure

```text
src/financial_volatility/
  benchmark/       Benchmark runner, walk-forward benchmark, hardware profiling
  config/          YAML settings and validation
  data/            OHLCV data types, loaders, splits
  evaluation/      Accuracy metrics and result records
  features/        Volatility features and sequence dataset construction
  models/          ForecastModel adapters and registry
  pipelines/       End-to-end local experiment pipeline
  results/         CSV result storage
  visualization/   Plots and thesis table generation
```

## Model Families

- Statistical: `garch`
- Statistical baseline: `linear_regression`
- Machine learning: `xgboost`
- Deep learning: `lstm`, `transformer`

Deep learning adapters use PyTorch sequence inputs with shape
`(samples, sequence_length, features)`.

The configured experiment pipeline automatically converts tabular supervised
features into sequence tensors when the selected model has a `sequence_length`
configuration.

## Outputs

Benchmark runs produce CSV metrics with accuracy and compute columns, including:

- `rmse`, `mae`, `mape`
- `training_time_seconds`, `inference_time_seconds`
- `peak_memory_mb`, `model_size_mb`

Visualization helpers can generate PNG plots and Markdown/CSV/LaTeX summary
tables from the stored results.
