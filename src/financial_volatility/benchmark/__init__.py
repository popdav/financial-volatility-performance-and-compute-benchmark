"""Compatibility exports for benchmark types."""

from financial_volatility.benchmark.hardware import (
    available_hardware_targets,
    is_cuda_available,
    resolve_torch_device,
)
from financial_volatility.benchmark.profiler import (
    PeakMemoryProfiler,
    estimate_model_size_mb,
)
from financial_volatility.benchmark.runner import BenchmarkRunner
from financial_volatility.benchmark.tenstorrent import (
    TenstorrentBenchmark,
    is_tenstorrent_available,
)
from financial_volatility.benchmark.types import (
    Forecast,
    HardwareTarget,
    MetadataValue,
    ModelInput,
    ModelMetadata,
    PredictionContext,
    ScalarValue,
)
from financial_volatility.benchmark.walk_forward import (
    WalkForwardBenchmark,
    WalkForwardBenchmarkResult,
)

__all__ = [
    "BenchmarkRunner",
    "Forecast",
    "HardwareTarget",
    "MetadataValue",
    "ModelInput",
    "ModelMetadata",
    "PeakMemoryProfiler",
    "PredictionContext",
    "ScalarValue",
    "TenstorrentBenchmark",
    "WalkForwardBenchmark",
    "WalkForwardBenchmarkResult",
    "available_hardware_targets",
    "estimate_model_size_mb",
    "is_cuda_available",
    "is_tenstorrent_available",
    "resolve_torch_device",
]
