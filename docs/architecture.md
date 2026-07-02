# Architecture

## 1. Project Purpose

This project is a Python benchmark platform for financial volatility forecasting. Its core thesis scope is to compare forecasting approaches across two dimensions:

- Forecast accuracy: how well each model predicts future volatility.
- Computational efficiency: how much time, memory, and compute each model requires.

The platform is intended to support statistical models, machine learning models, and deep learning models behind a common benchmark interface. The implementation should stay practical for a `src`-layout Python project and should prioritize reproducible local experiments over production infrastructure.

Kubernetes deployment, a Web UI, distributed execution, and hosted experiment tracking are future extensions. They are not part of the core thesis implementation.

## 2. High-Level Component Overview

The intended system is organized around a small number of clear components:

- Data ingestion: loads raw market data from local files or configured sources.
- Data validation: checks schema, date ordering, missing values, and required price columns.
- Feature preparation: converts market data into model-ready inputs such as returns, realized volatility, lagged values, or rolling statistics.
- Model adapters: wrap statistical, machine learning, and deep learning models behind the same `ForecastModel` interface.
- Benchmark runner: orchestrates datasets, train/test splits, model execution, metric calculation, and result writing.
- Metrics: calculates forecast accuracy metrics and computational efficiency metrics.
- Configuration: defines datasets, models, benchmark windows, metric choices, and output paths.
- Result storage: persists predictions, metrics, runtime metadata, and experiment configuration snapshots.
- Tests: verify interfaces, benchmark orchestration, metrics, configuration parsing, and result serialization.

The architecture should keep model-specific code isolated from benchmark orchestration. Adding a new model should not require changing the runner or metrics code unless the benchmark contract itself changes.

## 3. Proposed Directory Structure

The project should use a standard `src` layout:

```text
financial-volatility-performance-and-compute-benchmark/
├── configs/
│   ├── benchmark.yaml
│   ├── datasets.yaml
│   └── models.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── docs/
│   ├── architecture.md
│   └── roadmap.md
├── results/
│   ├── runs/
│   ├── metrics/
│   └── predictions/
├── src/
│   └── financial_volatility/
│       ├── __init__.py
│       ├── benchmark/
│       │   ├── runner.py
│       │   └── splits.py
│       ├── config/
│       │   ├── loader.py
│       │   ├── schema.py
│       │   └── settings.py
│       ├── data/
│       │   ├── loaders.py
│       │   └── types.py
│       ├── evaluation/
│       │   ├── metrics.py
│       │   └── results.py
│       ├── features/
│       │   └── engineering.py
│       ├── models/
│       │   ├── base.py
│       │   ├── statistical/
│       │   ├── machine_learning/
│       │   └── deep_learning/
│       └── results/
│           └── storage.py
├── tests/
│   ├── benchmark/
│   ├── config/
│   ├── data/
│   ├── metrics/
│   ├── models/
│   └── results/
├── pyproject.toml
└── README.md
```

`financial_volatility` is the canonical package namespace. Older experimental
namespaces should not be used for new code. The exact module names may evolve,
but the main separation should remain stable: data preparation, model adapters,
benchmark orchestration, metrics, configuration, and result storage.

## 4. Data Flow

The benchmark data flow should be explicit and reproducible:

1. Load benchmark configuration.
2. Load raw market data for one or more assets or datasets.
3. Validate required fields, date ordering, frequency, and missing values.
4. Transform prices into returns and volatility targets.
5. Create train, validation, and test windows using a configured split strategy.
6. Instantiate configured models through model adapters.
7. Fit each model on the training window.
8. Generate forecasts for the configured horizon.
9. Measure accuracy metrics from forecasts and observed volatility.
10. Measure computational metrics during fitting and prediction.
11. Store predictions, aggregate metrics, run metadata, and the resolved configuration.

For financial time series, split logic must avoid look-ahead bias. Walk-forward or rolling-window evaluation should be preferred over random splits.

## 5. Core Abstractions

The core abstractions should be small and testable:

- `ForecastModel`: common interface implemented by all model adapters.
- `DatasetSpec`: configuration describing a dataset location, symbol universe, date range, and required columns.
- `BenchmarkConfig`: resolved configuration for a full benchmark run.
- `SplitStrategy`: creates time-aware train/test windows.
- `Metric`: calculates one accuracy or compute metric from benchmark outputs.
- `BenchmarkResult`: structured output for one model, dataset, split, and forecast horizon.
- `ResultWriter`: writes results to local storage in stable formats.

These abstractions should use typed Python objects, such as dataclasses or typed configuration models, so configuration and result schemas can be validated early.

## 6. ForecastModel Interface

All statistical, machine learning, and deep learning models should be exposed through one interface. The interface should represent the benchmark contract rather than the internal API of any specific library.

Conceptually, each model should support:

- `name`: stable model identifier used in result files.
- `fit(train_data, validation_data=None)`: trains or estimates model parameters.
- `predict(horizon, context)`: returns volatility forecasts for the configured horizon.
- `get_params()`: returns model parameters needed for reproducibility.
- `supports_probabilistic_forecast`: declares whether the model can return uncertainty estimates.

Statistical models may wrap libraries such as ARCH/GARCH implementations. Machine learning models may wrap scikit-learn style estimators. Deep learning models may wrap PyTorch models. The adapter layer should absorb those differences so the benchmark runner only depends on the common interface.

The interface should not force every model family into identical internals. It should only standardize what the benchmark needs: fitting, forecasting, metadata, and reproducible parameters.

## 7. BenchmarkRunner Responsibilities

`BenchmarkRunner` should be the orchestration layer. It should not contain model-specific logic.

Its responsibilities should include:

- Loading and validating benchmark configuration.
- Loading datasets through the data layer.
- Building time-series splits.
- Instantiating configured model adapters.
- Running fit and predict steps for each model, dataset, split, and horizon.
- Capturing wall-clock runtime and optional memory usage.
- Calling metric functions for forecast accuracy and compute efficiency.
- Handling expected model failures in a structured way.
- Writing predictions, metrics, errors, and run metadata through `ResultWriter`.
- Producing deterministic run identifiers when possible.

The runner should be designed so a small local benchmark can run from the command line. Parallelism can be added later, but the core design should not require distributed infrastructure.

## 8. Configuration System

Configuration should be file-based and reproducible. YAML is a practical default because benchmark runs often need nested settings for datasets, model parameters, split strategies, horizons, and output paths.

Configuration should cover:

- Dataset paths and symbols.
- Date ranges and sampling frequency.
- Target volatility definition.
- Feature transformation settings.
- Train/test split strategy.
- Forecast horizons.
- Model list and model-specific parameters.
- Accuracy metrics.
- Compute metrics.
- Random seeds.
- Result output directory.

The loader should resolve multiple config files into one validated `BenchmarkConfig`. Each benchmark run should store a copy of the resolved configuration with the results.

## 9. Result Storage

The core thesis implementation should use local file-based result storage. This is enough for reproducible experiments and avoids unnecessary platform complexity.

Recommended outputs:

- Run metadata: JSON file containing timestamp, run ID, package version, machine details, and resolved config path.
- Metrics: CSV or Parquet table with one row per model, dataset, split, horizon, and metric.
- Predictions: CSV or Parquet table with forecast timestamps, predicted volatility, observed volatility, and model identifiers.
- Errors: JSON or JSONL file containing structured failures without stopping the entire benchmark when one model fails.
- Config snapshot: YAML or JSON copy of the resolved configuration used for the run.

Result files should be append-safe at the run level: each benchmark run writes to a new run directory under `results/runs/`.

## 10. Testing Strategy

Testing should focus on benchmark correctness and reproducibility rather than model performance.

Recommended test coverage:

- Unit tests for data validation and transformations.
- Unit tests for time-series split behavior, especially no look-ahead leakage.
- Unit tests for accuracy metrics on small known examples.
- Unit tests for compute metric collection with controlled dummy workloads.
- Contract tests for `ForecastModel` using simple fake models.
- Runner tests using tiny synthetic datasets and mock models.
- Configuration tests for valid and invalid benchmark config files.
- Result writer tests that verify schema, filenames, and reproducibility metadata.

Deep learning and heavier model tests should be kept separate from fast unit tests. The default test suite should stay lightweight enough to run in local development and CI.

## 11. Future Extensions

Future extensions can be added after the core thesis implementation is stable:

- Additional statistical models such as EGARCH, GJR-GARCH, and stochastic volatility models.
- Additional machine learning models such as gradient boosting, random forests, and support vector regression.
- Additional deep learning models such as LSTM, TCN, Transformer, and hybrid volatility models.
- Probabilistic forecasts and interval scoring.
- Multi-asset and portfolio-level volatility forecasting.
- Parallel local execution for independent model and split runs.
- Optional experiment tracking integrations.
- Optional containerized execution.
- Optional Kubernetes deployment for distributed benchmarks.
- Optional Web UI for browsing benchmark results.

Kubernetes and a Web UI should remain future work. They are useful platform features, but they are outside the core benchmark architecture needed for the thesis implementation.
