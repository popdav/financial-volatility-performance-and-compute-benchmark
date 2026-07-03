"""Sequence dataset construction for deep learning volatility models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd


@dataclass(frozen=True, slots=True)
class SequenceDataset:
    """Fixed-length feature sequences aligned to scalar targets."""

    X: npt.NDArray[np.float32]
    y: npt.NDArray[np.float32]
    target_timestamps: tuple[pd.Timestamp, ...]
    feature_columns: tuple[str, ...]


def build_sequence_dataset(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    sequence_length: int,
) -> SequenceDataset:
    """Convert aligned tabular features and targets into fixed-length sequences."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")

    if not isinstance(features.index, pd.DatetimeIndex):
        raise ValueError("features index must be a pandas DatetimeIndex")

    if not isinstance(target.index, pd.DatetimeIndex):
        raise ValueError("target index must be a pandas DatetimeIndex")

    aligned_features, aligned_target = _align_features_and_target(features, target)
    if len(aligned_features) < sequence_length:
        msg = (
            "sequence_length cannot exceed aligned data length: "
            f"{sequence_length} > {len(aligned_features)}"
        )
        raise ValueError(msg)

    feature_values = aligned_features.to_numpy(dtype=np.float32)
    target_values = aligned_target.to_numpy(dtype=np.float32)
    sequences: list[npt.NDArray[np.float32]] = []
    sequence_targets: list[np.float32] = []
    target_timestamps: list[pd.Timestamp] = []

    for target_position in range(sequence_length - 1, len(aligned_features)):
        window_start = target_position - sequence_length + 1
        sequences.append(feature_values[window_start : target_position + 1])
        sequence_targets.append(target_values[target_position])
        target_timestamps.append(aligned_target.index[target_position])

    return SequenceDataset(
        X=np.stack(sequences).astype(np.float32, copy=False),
        y=np.asarray(sequence_targets, dtype=np.float32),
        target_timestamps=tuple(target_timestamps),
        feature_columns=tuple(str(column) for column in aligned_features.columns),
    )


def _align_features_and_target(
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """Align feature rows and targets on common timestamps and drop missing values."""
    frame = features.join(target.rename("__target__"), how="inner")
    if frame.empty:
        raise ValueError("features and target must share at least one timestamp")

    frame = frame.sort_index().dropna()
    if frame.empty:
        raise ValueError("aligned features and target must contain no missing values")

    feature_frame = frame.drop(columns=["__target__"])
    target_series = frame["__target__"]

    if feature_frame.shape[1] == 0:
        raise ValueError("features must contain at least one column")

    if not np.all(np.isfinite(feature_frame.to_numpy(dtype=np.float64))):
        raise ValueError("features must be finite")

    if not np.all(np.isfinite(target_series.to_numpy(dtype=np.float64))):
        raise ValueError("target must be finite")

    return feature_frame, target_series


__all__ = ["SequenceDataset", "build_sequence_dataset"]
