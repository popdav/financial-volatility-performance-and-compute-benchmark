"""Hardware profiler tests."""

from pathlib import Path

from financial_volatility.benchmark.profiler import (
    PeakMemoryProfiler,
    estimate_model_size_mb,
)


class SavingModel:
    """Small persistable model used for size estimation tests."""

    def save(self, path: str | Path) -> None:
        """Write deterministic bytes to disk."""
        Path(path).write_bytes(b"model")


def test_peak_memory_profiler_records_non_negative_peak() -> None:
    """CPU memory profiling reports a non-negative peak allocation."""
    with PeakMemoryProfiler() as profiler:
        _values = [index for index in range(100)]
        peak_memory_mb = profiler.peak_memory_mb()

    assert peak_memory_mb >= 0.0


def test_model_size_estimation_uses_adapter_persistence() -> None:
    """Model size estimation measures serialized adapter output."""
    size_mb = estimate_model_size_mb(SavingModel())

    assert size_mb > 0.0
