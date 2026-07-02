"""Accuracy metric tests."""

import math

import pytest

from financial_volatility.evaluation.metrics import mae, mape, rmse


def test_rmse_returns_root_mean_squared_error() -> None:
    """RMSE is computed from array-like inputs."""
    assert rmse([1.0, 2.0, 3.0], [1.0, 2.0, 5.0]) == pytest.approx(math.sqrt(4.0 / 3.0))


def test_mae_returns_mean_absolute_error() -> None:
    """MAE is computed from array-like inputs."""
    assert mae((1.0, 2.0, 3.0), (2.0, 2.0, 1.0)) == pytest.approx(1.0)


def test_mape_returns_mean_absolute_percentage_error() -> None:
    """MAPE is reported as a percentage."""
    assert mape([100.0, 200.0], [110.0, 180.0]) == pytest.approx(10.0)


def test_mape_ignores_zero_true_values() -> None:
    """MAPE excludes zero true values to avoid division by zero."""
    assert mape([0.0, 100.0], [50.0, 110.0]) == pytest.approx(10.0)


def test_mape_returns_zero_when_all_true_and_predicted_values_are_zero() -> None:
    """All-zero true and predicted values have no percentage error."""
    assert mape([0.0, 0.0], [0.0, 0.0]) == 0.0


def test_mape_returns_infinity_when_all_true_values_are_zero_with_errors() -> None:
    """Non-zero predictions against all-zero true values are unbounded."""
    assert math.isinf(mape([0.0, 0.0], [0.0, 1.0]))


@pytest.mark.parametrize("metric", [rmse, mae, mape])
def test_metrics_reject_mismatched_shapes(metric: object) -> None:
    """Metrics require y_true and y_pred to have the same shape."""
    with pytest.raises(ValueError, match="same shape"):
        metric([1.0, 2.0], [1.0])


@pytest.mark.parametrize("metric", [rmse, mae, mape])
def test_metrics_reject_empty_inputs(metric: object) -> None:
    """Metrics require at least one observation."""
    with pytest.raises(ValueError, match="non-empty"):
        metric([], [])
