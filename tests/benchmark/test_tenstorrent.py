"""Optional Tenstorrent benchmark tests."""

import pytest

from financial_volatility.benchmark.tenstorrent import (
    TenstorrentBenchmark,
    is_tenstorrent_available,
)


def test_tenstorrent_availability_returns_boolean() -> None:
    """Tenstorrent detection has no hard runtime dependency."""
    assert isinstance(is_tenstorrent_available(), bool)


def test_tenstorrent_benchmark_skips_cleanly_without_runtime() -> None:
    """The optional benchmark fails clearly when no runtime is installed."""
    if is_tenstorrent_available():
        pytest.skip("Tenstorrent runtime is available in this environment")

    with pytest.raises(RuntimeError, match="not available"):
        TenstorrentBenchmark().run()
