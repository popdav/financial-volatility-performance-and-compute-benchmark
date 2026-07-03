"""CPU profiling helpers for benchmark runs."""

from __future__ import annotations

import pickle
import tempfile
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(slots=True)
class PeakMemoryProfiler:
    """Track peak Python memory allocations inside a benchmark scope."""

    _started: bool = False

    def __enter__(self) -> PeakMemoryProfiler:
        """Start memory tracing."""
        tracemalloc.start()
        self._started = True
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        """Stop memory tracing."""
        if self._started:
            tracemalloc.stop()
            self._started = False

    def peak_memory_mb(self) -> float:
        """Return peak traced memory in megabytes."""
        if not self._started:
            return 0.0

        _current, peak = tracemalloc.get_traced_memory()
        return peak / (1024.0 * 1024.0)


class PersistableModel(Protocol):
    """Subset of the model contract needed for size estimation."""

    def save(self, path: str | Path) -> None:
        """Persist model state."""


def estimate_model_size_mb(model: PersistableModel) -> float:
    """Estimate serialized model size in megabytes."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.bin"
        try:
            model.save(path)
        except Exception:
            return _pickle_size_mb(model)

        if path.exists():
            return path.stat().st_size / (1024.0 * 1024.0)

    return _pickle_size_mb(model)


def _pickle_size_mb(model: object) -> float:
    """Estimate object size via pickle when adapter persistence is unavailable."""
    try:
        return len(pickle.dumps(model)) / (1024.0 * 1024.0)
    except Exception:
        return 0.0


__all__ = ["PeakMemoryProfiler", "estimate_model_size_mb"]
