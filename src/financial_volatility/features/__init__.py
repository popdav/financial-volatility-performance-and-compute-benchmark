"""Feature engineering helpers for volatility forecasting."""

from financial_volatility.features.engineering import (
    add_lagged_returns,
    add_lagged_volatility,
    add_log_returns,
    add_moving_averages,
    add_realized_volatility,
    add_simple_returns,
    add_volume_change,
    build_supervised_dataset,
    build_volatility_features,
    make_volatility_target,
)
from financial_volatility.features.sequences import (
    SequenceDataset,
    build_sequence_dataset,
)

__all__ = [
    "SequenceDataset",
    "add_lagged_returns",
    "add_lagged_volatility",
    "add_log_returns",
    "add_moving_averages",
    "add_realized_volatility",
    "add_simple_returns",
    "add_volume_change",
    "build_sequence_dataset",
    "build_supervised_dataset",
    "build_volatility_features",
    "make_volatility_target",
]
