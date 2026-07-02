"""Framework-agnostic forecast accuracy metrics."""

import numpy as np
import numpy.typing as npt


def rmse(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    """Return root mean squared error for equally shaped non-empty inputs."""
    true_values, predicted_values = _validated_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean(np.square(true_values - predicted_values))))


def mae(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    """Return mean absolute error for equally shaped non-empty inputs."""
    true_values, predicted_values = _validated_arrays(y_true, y_pred)
    return float(np.mean(np.abs(true_values - predicted_values)))


def mape(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    """Return mean absolute percentage error, ignoring zero true values.

    Zero-valued ``y_true`` entries do not have a defined percentage error, so
    they are excluded from the calculation. If all true values are zero, the
    metric returns ``0.0`` when predictions are also all zero and ``inf``
    otherwise.
    """
    true_values, predicted_values = _validated_arrays(y_true, y_pred)
    non_zero_mask = true_values != 0

    if not np.any(non_zero_mask):
        if np.all(predicted_values == 0):
            return 0.0
        return float("inf")

    percentage_errors = np.abs(
        (true_values[non_zero_mask] - predicted_values[non_zero_mask])
        / true_values[non_zero_mask]
    )
    return float(np.mean(percentage_errors) * 100.0)


def _validated_arrays(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Convert metric inputs to float arrays and validate their shape."""
    true_values = np.asarray(y_true, dtype=np.float64)
    predicted_values = np.asarray(y_pred, dtype=np.float64)

    if true_values.shape != predicted_values.shape:
        msg = (
            "y_true and y_pred must have the same shape: "
            f"{true_values.shape} != {predicted_values.shape}"
        )
        raise ValueError(msg)

    if true_values.size == 0:
        raise ValueError("y_true and y_pred must be non-empty")

    return true_values, predicted_values
