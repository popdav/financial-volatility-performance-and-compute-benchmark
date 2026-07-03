"""Compatibility exports for model contracts."""

from financial_volatility.models.base import ForecastModel
from financial_volatility.models.garch import GARCHModel
from financial_volatility.models.linear import LinearRegressionModel
from financial_volatility.models.lstm import LSTMConfig, LSTMModel
from financial_volatility.models.registry import ModelRegistry, register_default_models
from financial_volatility.models.transformer import TransformerConfig, TransformerModel
from financial_volatility.models.xgboost import XGBoostModel

__all__ = [
    "ForecastModel",
    "GARCHModel",
    "LSTMConfig",
    "LSTMModel",
    "LinearRegressionModel",
    "ModelRegistry",
    "TransformerConfig",
    "TransformerModel",
    "XGBoostModel",
    "register_default_models",
]
