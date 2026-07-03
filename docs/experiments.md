# Experiment Documentation

## Dataset

Experiments consume OHLCV market data with a `DatetimeIndex` and the required
columns `open`, `high`, `low`, `close`, and `volume`.

The local experiment pipeline currently supports CSV input through
`load_ohlcv_csv`. The CLI can also generate a deterministic synthetic OHLCV CSV
for smoke-test runs when no local path is supplied.

## Features

The default tabular feature set is built by `build_volatility_features` and
includes:

- simple one-day return
- one-day log return
- rolling realized volatility over 5 and 21 days
- return lags over 1 and 5 days
- one-day volatility lag
- moving averages over 5 and 21 days

Sequence models use `build_sequence_dataset` to convert aligned tabular features
and targets into arrays shaped `(samples, sequence_length, features)`.

## Targets

The default supervised target is a realized volatility column such as
`realized_volatility_5d`. Target construction must remain shifted and aligned so
feature rows do not include future observations.

## Models

Implemented adapters:

- `garch`: statistical GARCH(p, q) model using `arch`
- `linear_regression`: scikit-learn linear baseline
- `xgboost`: tabular machine learning model using `XGBRegressor`
- `lstm`: PyTorch sequence model
- `transformer`: PyTorch Transformer encoder sequence model

All adapters implement the shared `ForecastModel` contract.

## Metrics

Accuracy metrics:

- RMSE
- MAE
- MAPE

Compute metrics:

- training time
- inference time
- peak Python memory allocation
- serialized model size

## Reproducibility

Experiments are configured with YAML files, deterministic synthetic test data,
typed model metadata, and stable CSV result schemas. Time-series splitting is
chronological, and walk-forward evaluation retrains a fresh model per fold.
