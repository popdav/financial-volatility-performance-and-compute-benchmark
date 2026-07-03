"""Sequence dataset builder tests."""

import numpy as np
import pandas as pd
import pytest

from financial_volatility.features.sequences import (
    SequenceDataset,
    build_sequence_dataset,
)


def test_sequence_dataset_shape_is_correct() -> None:
    """Feature rows are converted to fixed-length 3D sequences."""
    dataset = build_sequence_dataset(
        _features(),
        _target(),
        sequence_length=4,
    )

    assert isinstance(dataset, SequenceDataset)
    assert dataset.X.shape == (7, 4, 2)
    assert dataset.y.shape == (7,)
    assert dataset.X.dtype == np.float32
    assert dataset.y.dtype == np.float32


def test_sequence_dataset_preserves_target_alignment() -> None:
    """Each target aligns with the final timestamp in its input window."""
    features = _features()
    target = _target()

    dataset = build_sequence_dataset(features, target, sequence_length=3)

    assert dataset.target_timestamps[0] == pd.Timestamp("2026-01-03")
    assert dataset.y[0] == pytest.approx(target.loc["2026-01-03"])
    assert dataset.X[0, :, 0].tolist() == pytest.approx([0.0, 1.0, 2.0])
    assert dataset.X[0, -1, 0] == pytest.approx(features.loc["2026-01-03", "x1"])


def test_sequence_dataset_aligns_on_common_index() -> None:
    """Mismatched feature and target ranges are aligned by timestamp."""
    features = _features().iloc[1:]
    target = _target().iloc[:-1]

    dataset = build_sequence_dataset(features, target, sequence_length=2)

    assert dataset.target_timestamps[0] == pd.Timestamp("2026-01-03")
    assert dataset.target_timestamps[-1] == pd.Timestamp("2026-01-09")


def test_sequence_dataset_rejects_invalid_inputs() -> None:
    """Sequence construction validates indexes, lengths, and missing values."""
    with pytest.raises(ValueError, match="sequence_length"):
        build_sequence_dataset(_features(), _target(), sequence_length=0)

    with pytest.raises(ValueError, match="DatetimeIndex"):
        build_sequence_dataset(
            _features().reset_index(drop=True),
            _target(),
            sequence_length=2,
        )

    with pytest.raises(ValueError, match="cannot exceed"):
        build_sequence_dataset(_features(rows=2), _target(rows=2), sequence_length=3)

    features = _features()
    features.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        build_sequence_dataset(features.iloc[:1], _target(rows=1), sequence_length=1)


def _features(rows: int = 10) -> pd.DataFrame:
    """Create deterministic tabular features."""
    index = pd.date_range("2026-01-01", periods=rows, freq="D")
    return pd.DataFrame(
        {
            "x1": np.arange(rows, dtype=np.float64),
            "x2": np.arange(rows, dtype=np.float64) + 10.0,
        },
        index=index,
    )


def _target(rows: int = 10) -> pd.Series:
    """Create deterministic targets."""
    index = pd.date_range("2026-01-01", periods=rows, freq="D")
    return pd.Series(
        np.arange(rows, dtype=np.float64) + 100.0,
        index=index,
        name="target",
    )
