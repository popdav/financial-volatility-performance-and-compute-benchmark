# Roadmap

## Current Thesis Scope

The repository now implements the core local benchmark platform described in
`docs/architecture.md`:

- provider-agnostic OHLCV containers and validation
- local CSV loading
- Yahoo Finance loading with mocked test coverage
- deterministic Parquet caching
- volatility feature engineering and future target construction
- chronological and walk-forward splitting
- ForecastModel adapters for GARCH, linear regression, XGBoost, LSTM, and a
  simple Transformer encoder
- benchmark timing, memory, model-size, RMSE, MAE, and MAPE metrics
- YAML configuration and model registry
- Typer CLI for validation, model listing, and experiment runs
- CSV result persistence, plots, and deterministic thesis tables

The default reproducible experiment is:

```bash
uv run fvbench run --config configs/default.yaml
```

It uses deterministic synthetic OHLCV data, `linear_regression`, CPU execution,
and a 5-day future realized-volatility target.

## Near-Term Stabilization

- Run small real local-CSV experiments for each registered model and compare
  outputs manually before running thesis-scale datasets.
- Add representative checked-in sample configs for each model family.
- Decide whether the thesis will use the simple Transformer implementation as
  the final "Transformer/PatchTST" model or replace it with a true PatchTST
  adapter.
- Add persistence for per-timestamp prediction files if forecast inspection is
  required in the thesis results chapter.

## Out of Scope Until Thesis Experiments Are Stable

- Kubernetes deployment
- Web UI
- Hosted experiment tracking
- Distributed execution
- Online inference services
