# Financial Volatility Benchmark Platform — Detailed Development Backlog

## Project Objective

Build a modular Python research platform for benchmarking statistical, machine learning, and deep learning models for financial volatility forecasting.

The platform should support:

* reproducible experiments,
* unified model interface,
* consistent evaluation metrics,
* computational efficiency measurement,
* hardware-aware benchmarking,
* thesis-ready tables and plots.

Do not implement future deployment features until the thesis benchmark is complete.

---

# Phase 1 — Project Foundation

Status: Completed

Expected completed items:

* uv project setup
* src layout
* pyproject.toml
* Ruff
* pytest
* mypy
* pre-commit
* basic package structure
* docs/architecture.md
* docs/roadmap.md

---

# Phase 2 — Data Layer

## Task 2.1 — Market Data Types

Status: Completed

Goal:

Create provider-agnostic data structures used by all loaders and models.

Deliverables:

* `OHLCVData`
* `TimeSeriesDataset`
* `TrainTestSplit`

Acceptance criteria:

* validates DatetimeIndex
* validates chronological order
* validates required columns
* rejects empty data
* tests pass

---

## Task 2.2 — Local CSV Loader

Status: Completed

Goal:

Load historical OHLCV data from a local CSV file.

Deliverables:

* `src/financial_volatility/data/loaders.py`
* tests using synthetic CSV files

Requirements:

* configurable date column
* lowercase column normalization
* date column converted to DatetimeIndex
* data sorted chronologically
* returns `OHLCVData`

Acceptance criteria:

* valid CSV loads
* missing date column fails
* invalid dates fail
* missing OHLCV columns fail

---

## Task 2.3 — Yahoo Finance Downloader

Status: Completed

Goal:

Download OHLCV data from Yahoo Finance.

Deliverables:

* `src/financial_volatility/data/yahoo.py`
* tests with mocked downloader
* optional cache integration

Requirements:

* configurable symbol
* configurable start/end dates
* returns `OHLCVData`
* handles adjusted close if available
* logs download activity
* avoids network calls in tests

Acceptance criteria:

* downloader interface works
* mocked response converts to `OHLCVData`
* invalid symbol or empty response raises clear error

---

## Task 2.4 — Parquet Cache

Status: Completed

Goal:

Avoid repeated downloads and make experiments reproducible.

Deliverables:

* `src/financial_volatility/data/cache.py`

Requirements:

* save OHLCV data to parquet
* load OHLCV data from parquet
* deterministic cache path based on provider, symbol, dates
* create cache directories automatically

Acceptance criteria:

* cache write/read roundtrip works
* missing cache path handled cleanly
* tests use tmp_path

---

# Phase 3 — Feature Engineering

## Task 3.1 — Basic Volatility Features

Status: Completed

Goal:

Transform OHLCV data into a supervised learning dataset.

Deliverables:

* `src/financial_volatility/features/engineering.py`

Features:

* simple return
* log return
* rolling realized volatility 5d
* rolling realized volatility 21d
* return lag 1
* return lag 5
* volatility lag 1
* moving average 5
* moving average 21
* volume change

Acceptance criteria:

* synthetic data tests pass
* lag features are shifted correctly
* rolling volatility uses log returns
* missing rows caused by lags/windows are dropped

---

## Task 3.2 — Target Construction

Status: Completed

Goal:

Create forecasting target for realized volatility.

Deliverables:

* `make_volatility_target(...)`

Requirements:

* support horizon = 1, 5, 21
* target should represent future realized volatility
* target must be shifted so there is no lookahead bias
* output aligned with feature matrix

Acceptance criteria:

* horizon 1 works
* horizon 5 works
* horizon 21 works
* no target uses future data in features
* tests verify alignment

---

## Task 3.3 — Feature Matrix Builder

Status: Completed

Goal:

Produce clean `X` and `y`.

Deliverables:

* `build_supervised_dataset(...)`

Requirements:

* input: `OHLCVData`
* output: `X`, `y`
* configurable forecast horizon
* configurable rolling windows
* configurable lags
* drop missing values after alignment

Acceptance criteria:

* X and y have equal length
* no NaN values
* DatetimeIndex preserved
* tests pass

---

# Phase 4 — Time-Series Splitting

## Task 4.1 — Chronological Train/Test Split

Status: Completed

Goal:

Split data without shuffling.

Requirements:

* split by float test_size
* split by split_date
* preserve chronological order
* reject empty train/test sets

Acceptance criteria:

* float split works
* date split works
* invalid split fails

---

## Task 4.2 — Walk-Forward Splitter

Status: Completed

Goal:

Implement realistic financial time-series evaluation.

Deliverables:

* `src/financial_volatility/data/walk_forward.py`

Requirements:

* expanding window mode
* optional rolling window mode
* configurable initial train size
* configurable test window size
* configurable step size
* iterator interface

Acceptance criteria:

* produces chronological folds
* no overlap mistakes
* no lookahead
* tests verify fold boundaries

---

# Phase 5 — Benchmark Framework

## Task 5.1 — BenchmarkRunner Improvements

Status: Completed

Goal:

Ensure all models are evaluated consistently.

Metrics:

* RMSE
* MAE
* MAPE
* training time
* inference time
* peak memory
* model size

Requirements:

* works with any `ForecastModel`
* returns `ExperimentResult`
* no model-specific logic

Acceptance criteria:

* dummy model works
* timing values are non-negative
* metrics match expected values

---

## Task 5.2 — Result Storage

Status: Completed

Goal:

Persist benchmark results.

Deliverables:

* CSV writer
* CSV append
* stable schema

Columns:

* experiment_id
* model_name
* model_type
* hardware
* dataset
* horizon
* rmse
* mae
* mape
* training_time_seconds
* inference_time_seconds
* peak_memory_mb
* model_size_mb
* timestamp

---

# Phase 6 — Experiment Pipeline

## Task 6.1 — Minimal Experiment Pipeline

Status: Completed

Goal:

Run one complete experiment end-to-end.

Pipeline:

```text
CSV/Yahoo data
→ feature engineering
→ target construction
→ train/test split
→ model training
→ prediction
→ benchmark
→ CSV results
```

Requirements:

* model is passed in
* no concrete model selection yet
* use dummy model in tests

Acceptance criteria:

* pipeline runs with synthetic CSV
* results CSV is produced
* metrics are calculated

---

## Task 6.2 — Configuration-Driven Experiment

Status: Completed

Goal:

Run experiments from YAML.

Requirements:

* load config
* select dataset source
* select model name
* select horizon
* select output directory
* pass model parameters

Acceptance criteria:

* config-driven dummy experiment works
* invalid config fails clearly

---

# Phase 7 — Model Registry

## Task 7.1 — Model Registry

Status: Completed

Goal:

Avoid if/elif chains for model selection.

Deliverables:

* `src/financial_volatility/models/registry.py`

Requirements:

* register model classes by name
* instantiate model from config
* support parameters

Example:

```python
model = ModelRegistry.create("xgboost", parameters={"max_depth": 3})
```

Acceptance criteria:

* registered model can be created
* unknown model raises clear error
* tests use dummy model

---

# Phase 8 — Statistical Models

## Task 8.1 — GARCH(1,1)

Status: Completed

Goal:

Implement first real financial model.

Requirements:

* use `arch`
* implement `ForecastModel`
* train on returns
* forecast realized volatility
* metadata:

  * name: garch
  * type: statistical
  * parameters: p=1, q=1

Acceptance criteria:

* trains on synthetic returns
* predicts non-empty output
* integrates with BenchmarkRunner

---

## Task 8.2 — GARCH Walk-Forward Evaluation

Status: Completed

Goal:

Evaluate GARCH in a finance-realistic way.

Requirements:

* retrain per fold
* aggregate fold metrics
* store fold-level and aggregate results

Acceptance criteria:

* walk-forward benchmark runs
* fold results saved
* aggregate metrics saved

---

# Phase 9 — Classical Machine Learning Models

## Task 9.1 — Linear Regression

Status: Completed

Goal:

Implement simple statistical/ML baseline.

Requirements:

* use scikit-learn
* implement `ForecastModel`
* train on engineered features
* save/load with joblib
* metadata:

  * name: linear_regression
  * type: statistical_baseline

Acceptance criteria:

* trains on synthetic features
* predicts correct length
* save/load works
* benchmark integration works

---

## Task 9.2 — XGBoost

Status: Completed

Goal:

Implement practical tabular ML model.

Requirements:

* use `xgboost.XGBRegressor`
* implement `ForecastModel`
* configurable hyperparameters
* save/load
* metadata:

  * name: xgboost
  * type: machine_learning

Acceptance criteria:

* trains on synthetic features
* predicts correct length
* save/load works
* benchmark integration works
* no duplicated benchmark logic

---

# Phase 10 — Deep Learning Preparation

## Task 10.1 — Sequence Dataset Builder

Status: Completed

Goal:

Prepare inputs for LSTM and Transformer.

Requirements:

* convert tabular features into sequences
* configurable sequence length
* preserve target alignment
* output suitable for PyTorch

Example:

```text
previous 60 days of features → target volatility
```

Acceptance criteria:

* sequence shape is correct
* target alignment is correct
* no lookahead bias
* tests pass

---

# Phase 11 — Deep Learning Models

## Task 11.1 — LSTM

Status: Completed

Goal:

Implement sequence-based deep learning baseline.

Requirements:

* use PyTorch
* implement `ForecastModel`
* configurable:

  * sequence length
  * hidden size
  * number of layers
  * dropout
  * epochs
  * learning rate
* save/load model weights

Acceptance criteria:

* trains on synthetic sequence data
* predicts correct length
* benchmark integration works
* supports CPU/GPU device parameter

---

## Task 11.2 — Transformer / PatchTST

Status: Completed with deviation

Implemented as a compact PyTorch Transformer encoder adapter, not a full
PatchTST architecture.

Goal:

Implement modern time-series deep learning model.

Requirements:

* choose simple Transformer or PatchTST
* use PyTorch
* implement `ForecastModel`
* configurable:

  * sequence length
  * model dimension
  * number of heads
  * number of layers
  * dropout
  * epochs
  * learning rate
* save/load weights

Acceptance criteria:

* trains on synthetic sequence data
* predicts correct length
* benchmark integration works
* supports CPU/GPU device parameter

---

# Phase 12 — Hardware Benchmarking

## Task 12.1 — Hardware Profiler

Status: Completed

Goal:

Measure computational efficiency.

Metrics:

* wall-clock training time
* wall-clock inference time
* peak memory
* model size
* optional throughput

Requirements:

* CPU support required
* GPU support optional
* no Tenstorrent dependency yet

Acceptance criteria:

* profiler works on CPU
* integrates with BenchmarkRunner
* tests pass without GPU

---

## Task 12.2 — GPU Benchmarking

Status: Completed for graceful optional handling

Goal:

Compare CPU and GPU behavior where applicable.

Requirements:

* detect CUDA availability
* skip GPU tests if unavailable
* support device config
* benchmark LSTM/Transformer on GPU if available

Acceptance criteria:

* CPU-only environment still passes
* GPU results stored when available

---

## Task 12.3 — Optional Tenstorrent Benchmark

Status: Completed as optional detection/documentation

Goal:

Explore hardware-aware deployment if feasible.

Requirements:

* optional module
* no hard dependency
* graceful skip if hardware/toolchain unavailable

Acceptance criteria:

* project still works without Tenstorrent hardware
* TT experiment documented as optional

---

# Phase 13 — Visualization

## Task 13.1 — Result Plots

Status: Completed

Goal:

Generate thesis-ready plots.

Plots:

* RMSE by model
* MAE by model
* training time by model
* inference time by model
* memory by model
* accuracy vs computational cost scatter plot

Requirements:

* use matplotlib
* save PNG files
* no seaborn required

Acceptance criteria:

* plots generated from results CSV
* tests verify files are created

---

## Task 13.2 — Thesis Tables

Status: Completed

Goal:

Generate clean tables for thesis.

Outputs:

* Markdown table
* CSV summary
* optional LaTeX table

Acceptance criteria:

* table includes accuracy and compute metrics
* output deterministic

---

# Phase 14 — CLI

## Task 14.1 — Typer CLI Foundation

Status: Completed

Goal:

Run experiments from terminal.

Example:

```bash
uv run fvbench run --config configs/xgboost.yaml
```

Commands:

* `fvbench run`
* `fvbench validate-config`
* `fvbench list-models`

Acceptance criteria:

* CLI help works
* config validation works
* real synthetic experiment can run

---

## Task 14.2 — Real Experiment CLI

Status: Completed

Goal:

Run real models from config.

Requirements:

* use ModelRegistry
* load selected model
* run ExperimentPipeline
* save results
* print summary

Acceptance criteria:

* GARCH config runs
* Linear config runs
* XGBoost config runs after XGBoost is implemented

---

# Phase 15 — Documentation

## Task 15.1 — README Update

Status: Completed

Add:

* project purpose
* setup instructions
* running tests
* running experiments
* project structure

---

## Task 15.2 — Experiment Documentation

Status: Completed

Add:

* dataset description
* feature description
* model list
* metric definitions
* reproducibility notes

---

## Task 15.3 — Thesis Mapping

Status: Completed

Create `docs/thesis-mapping.md`.

Map code modules to thesis chapters.

Example:

* `features/engineering.py` → Methodology
* `models/garch.py` → Statistical Models
* `benchmark/runner.py` → Experimental Setup
* `results/` → Results chapter

---

# Phase 16 — Future Work

Do not implement before thesis experiments are complete.

Possible extensions:

* FastAPI model serving
* Docker
* Kubernetes
* Web UI
* Prometheus
* Grafana
* online inference demo
* model comparison dashboard

---

# Global Acceptance Criteria

Every task must satisfy:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pre-commit run --all-files
```

Every task summary must include:

1. What was implemented
2. Files created
3. Files modified
4. Tests added
5. Commands run
6. Next recommended task

---

# Implementation Rule

Do not implement future phases early.

For example:

* Do not implement XGBoost while working on data loaders.
* Do not implement Kubernetes before thesis experiments.
* Do not add GPU-specific code before CPU benchmarking works.
* Do not add UI until all thesis models are benchmarked.

The priority is always:

```text
working thesis experiment > perfect platform
```
