"""Walk-forward splitting tests."""

import pandas as pd
import pytest

from financial_volatility.data.types import TimeSeriesDataset
from financial_volatility.data.walk_forward import (
    WalkForwardSplitter,
    walk_forward_split,
)


def test_expanding_walk_forward_split_produces_chronological_folds() -> None:
    """Expanding folds grow the train set and keep test rows in the future."""
    folds = list(
        walk_forward_split(
            _frame(),
            initial_train_size=5,
            test_window_size=2,
            step_size=2,
            name="expanding",
        )
    )

    assert len(folds) == 3
    assert folds[0].train.row_count == 5
    assert folds[0].test.row_count == 2
    assert folds[1].train.row_count == 7
    assert folds[2].train.row_count == 9
    assert folds[0].train.end_timestamp < folds[0].test.start_timestamp
    assert folds[2].test.end_timestamp == pd.Timestamp("2026-01-11")


def test_rolling_walk_forward_split_keeps_fixed_train_window() -> None:
    """Rolling folds use only the configured trailing train observations."""
    folds = list(
        walk_forward_split(
            _frame(),
            initial_train_size=5,
            test_window_size=2,
            step_size=2,
            rolling_window_size=4,
        )
    )

    assert [fold.train.row_count for fold in folds] == [4, 4, 4]
    assert folds[0].train.start_timestamp == pd.Timestamp("2026-01-02")
    assert folds[1].train.start_timestamp == pd.Timestamp("2026-01-04")
    assert folds[2].train.start_timestamp == pd.Timestamp("2026-01-06")


def test_walk_forward_splitter_instance_is_iterable_factory() -> None:
    """Splitter instances can produce folds via the shared split method."""
    splitter = WalkForwardSplitter(
        initial_train_size=4,
        test_window_size=3,
        step_size=3,
    )

    folds = list(splitter.split(_frame(rows=10), name="wf"))

    assert [fold.name for fold in folds] == ["wf_fold_0", "wf_fold_1"]
    assert folds[0].test.start_timestamp == pd.Timestamp("2026-01-05")
    assert folds[1].test.start_timestamp == pd.Timestamp("2026-01-08")


def test_walk_forward_split_sorts_input_before_splitting() -> None:
    """Unsorted input is sorted chronologically before folds are generated."""
    unsorted = _frame().sort_index(ascending=False)

    fold = next(
        walk_forward_split(
            unsorted,
            initial_train_size=5,
            test_window_size=2,
        )
    )

    assert fold.train.to_dataframe().index.is_monotonic_increasing
    assert fold.test.to_dataframe().index.is_monotonic_increasing
    assert fold.train.end_timestamp < fold.test.start_timestamp


def test_walk_forward_split_supports_time_series_dataset_input() -> None:
    """TimeSeriesDataset instances can be split directly."""
    dataset = TimeSeriesDataset(data=_frame(), name="features")

    folds = list(
        walk_forward_split(
            dataset,
            initial_train_size=5,
            test_window_size=2,
            step_size=2,
        )
    )

    assert len(folds) == 3
    assert folds[0].train.name == "walk_forward_fold_0_train"


def test_walk_forward_split_rejects_invalid_configuration() -> None:
    """Window sizes and step sizes must be positive."""
    with pytest.raises(ValueError, match="initial_train_size"):
        WalkForwardSplitter(initial_train_size=0, test_window_size=1)

    with pytest.raises(ValueError, match="test_window_size"):
        WalkForwardSplitter(initial_train_size=1, test_window_size=0)

    with pytest.raises(ValueError, match="step_size"):
        WalkForwardSplitter(initial_train_size=1, test_window_size=1, step_size=0)

    with pytest.raises(ValueError, match="rolling_window_size"):
        WalkForwardSplitter(
            initial_train_size=1,
            test_window_size=1,
            rolling_window_size=0,
        )


def test_walk_forward_split_rejects_invalid_input() -> None:
    """Input must be non-empty and time-indexed."""
    with pytest.raises(ValueError, match="non-empty"):
        list(
            walk_forward_split(
                _frame(rows=0),
                initial_train_size=1,
                test_window_size=1,
            )
        )

    with pytest.raises(ValueError, match="DatetimeIndex"):
        list(
            walk_forward_split(
                _frame().reset_index(drop=True),
                initial_train_size=1,
                test_window_size=1,
            )
        )


def test_walk_forward_split_returns_no_partial_folds() -> None:
    """Trailing rows that cannot fill a test window are not emitted."""
    folds = list(
        walk_forward_split(
            _frame(rows=8),
            initial_train_size=5,
            test_window_size=2,
            step_size=2,
        )
    )

    assert len(folds) == 1
    assert folds[0].test.end_timestamp == pd.Timestamp("2026-01-07")


def _frame(rows: int = 12) -> pd.DataFrame:
    """Create synthetic time-indexed data."""
    index = pd.date_range("2026-01-01", periods=rows, freq="D")
    return pd.DataFrame({"value": list(range(rows))}, index=index)
