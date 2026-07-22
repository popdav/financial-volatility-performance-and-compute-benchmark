"""Feature engineering helpers for volatility forecasting."""

from financial_volatility.features.engineering import (
    DatasetValidationReport,
    FeatureConfig,
    FeatureMetadata,
    add_atr,
    add_exponential_moving_averages,
    add_lagged_returns,
    add_lagged_volatility,
    add_log_returns,
    add_moving_averages,
    add_realized_volatility,
    add_rsi,
    add_simple_returns,
    add_volume_change,
    add_volume_features,
    build_feature_matrix,
    build_supervised_dataset,
    build_volatility_features,
    make_volatility_target,
    validate_supervised_dataset,
)
from financial_volatility.features.inspection import generate_feature_inspection
from financial_volatility.features.sequences import (
    SequenceDataset,
    build_sequence_dataset,
)

__all__ = [
    "DatasetValidationReport",
    "FeatureConfig",
    "FeatureMetadata",
    "SequenceDataset",
    "add_atr",
    "add_exponential_moving_averages",
    "add_lagged_returns",
    "add_lagged_volatility",
    "add_log_returns",
    "add_moving_averages",
    "add_realized_volatility",
    "add_rsi",
    "add_simple_returns",
    "add_volume_change",
    "add_volume_features",
    "build_feature_matrix",
    "build_sequence_dataset",
    "build_supervised_dataset",
    "build_volatility_features",
    "generate_feature_inspection",
    "make_volatility_target",
    "validate_supervised_dataset",
]
