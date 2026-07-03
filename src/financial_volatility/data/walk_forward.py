"""Walk-forward splitting utilities for financial time-series evaluation."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd

from financial_volatility.data.types import TimeSeriesDataset, TrainTestSplit


@dataclass(frozen=True, slots=True)
class WalkForwardSplitter:
    """Generate chronological train/test folds for walk-forward evaluation."""

    initial_train_size: int
    test_window_size: int
    step_size: int = 1
    rolling_window_size: int | None = None

    def __post_init__(self) -> None:
        """Validate splitter configuration."""
        if self.initial_train_size <= 0:
            raise ValueError("initial_train_size must be positive")

        if self.test_window_size <= 0:
            raise ValueError("test_window_size must be positive")

        if self.step_size <= 0:
            raise ValueError("step_size must be positive")

        if self.rolling_window_size is not None and self.rolling_window_size <= 0:
            raise ValueError("rolling_window_size must be positive")

    def split(
        self,
        data: pd.DataFrame | TimeSeriesDataset,
        *,
        name: str = "walk_forward",
    ) -> Iterator[TrainTestSplit]:
        """Yield chronological walk-forward folds."""
        frame = _as_sorted_dataframe(data)
        if frame.empty:
            raise ValueError("Input data must be non-empty")

        fold_index = 0
        test_start = self.initial_train_size
        while test_start + self.test_window_size <= len(frame):
            train_start = _train_start(
                test_start=test_start,
                rolling_window_size=self.rolling_window_size,
            )
            train_frame = frame.iloc[train_start:test_start]
            test_frame = frame.iloc[test_start : test_start + self.test_window_size]

            if train_frame.empty or test_frame.empty:
                raise ValueError("Walk-forward split produced an empty fold")

            yield TrainTestSplit(
                train=TimeSeriesDataset(
                    data=train_frame,
                    name=f"{name}_fold_{fold_index}_train",
                ),
                test=TimeSeriesDataset(
                    data=test_frame,
                    name=f"{name}_fold_{fold_index}_test",
                ),
                name=f"{name}_fold_{fold_index}",
            )

            fold_index += 1
            test_start += self.step_size

    def __call__(
        self,
        data: pd.DataFrame | TimeSeriesDataset,
        *,
        name: str = "walk_forward",
    ) -> Iterator[TrainTestSplit]:
        """Allow splitter instances to be used as iterator factories."""
        return self.split(data, name=name)


def walk_forward_split(
    data: pd.DataFrame | TimeSeriesDataset,
    *,
    initial_train_size: int,
    test_window_size: int,
    step_size: int = 1,
    rolling_window_size: int | None = None,
    name: str = "walk_forward",
) -> Iterator[TrainTestSplit]:
    """Convenience wrapper returning walk-forward train/test folds."""
    splitter = WalkForwardSplitter(
        initial_train_size=initial_train_size,
        test_window_size=test_window_size,
        step_size=step_size,
        rolling_window_size=rolling_window_size,
    )
    return splitter.split(data, name=name)


def _as_sorted_dataframe(data: pd.DataFrame | TimeSeriesDataset) -> pd.DataFrame:
    """Return a chronological DataFrame copy from supported split inputs."""
    if isinstance(data, TimeSeriesDataset):
        frame = data.to_dataframe()
    else:
        frame = data.copy(deep=True)

    if not isinstance(frame.index, pd.DatetimeIndex):
        msg = "Input data index must be a pandas DatetimeIndex"
        raise ValueError(msg)

    return frame.sort_index()


def _train_start(test_start: int, rolling_window_size: int | None) -> int:
    """Calculate the train window start for expanding or rolling folds."""
    if rolling_window_size is None:
        return 0

    return max(0, test_start - rolling_window_size)


__all__ = ["WalkForwardSplitter", "walk_forward_split"]
