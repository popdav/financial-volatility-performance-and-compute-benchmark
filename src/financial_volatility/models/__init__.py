"""Compatibility exports for model contracts."""

from financial_volatility.models.base import ForecastModel
from financial_volatility.models.garch import GARCHModel
from financial_volatility.models.linear import LinearRegressionModel

__all__ = ["ForecastModel", "GARCHModel", "LinearRegressionModel"]
