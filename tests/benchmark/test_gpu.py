"""GPU benchmark support tests."""

import numpy as np
import pytest

from financial_volatility.benchmark import (
    BenchmarkRunner,
    HardwareTarget,
    ModelInput,
    PredictionContext,
    available_hardware_targets,
    is_cuda_available,
)
from financial_volatility.models import LSTMModel, TransformerModel


def test_available_hardware_targets_always_include_cpu() -> None:
    """CPU-only environments remain valid benchmark targets."""
    assert HardwareTarget.CPU in available_hardware_targets()


@pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available")
def test_lstm_gpu_benchmark_runs_when_cuda_is_available() -> None:
    """LSTM can be benchmarked on CUDA when the device is present."""
    features, target = _sequence_data()

    result = BenchmarkRunner(
        model=LSTMModel(
            sequence_length=4,
            hidden_size=4,
            epochs=1,
            device="cuda",
        ),
        training_data=ModelInput(features=features[:16], target=target[:16]),
        test_input_data=PredictionContext(features=features[16:18]),
        test_target_data=target[16:18],
        hardware_target=HardwareTarget.CUDA,
    ).run()

    assert result.hardware == HardwareTarget.CUDA
    assert result.forecast is not None


@pytest.mark.skipif(not is_cuda_available(), reason="CUDA is not available")
def test_transformer_gpu_benchmark_runs_when_cuda_is_available() -> None:
    """Transformer can be benchmarked on CUDA when the device is present."""
    features, target = _sequence_data()

    result = BenchmarkRunner(
        model=TransformerModel(
            sequence_length=4,
            model_dimension=4,
            num_heads=2,
            epochs=1,
            device="cuda",
        ),
        training_data=ModelInput(features=features[:16], target=target[:16]),
        test_input_data=PredictionContext(features=features[16:18]),
        test_target_data=target[16:18],
        hardware_target=HardwareTarget.CUDA,
    ).run()

    assert result.hardware == HardwareTarget.CUDA
    assert result.forecast is not None


def _sequence_data() -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic synthetic sequence data."""
    rng = np.random.default_rng(42)
    features = rng.normal(size=(20, 4, 3)).astype(np.float32)
    target = (features[:, -1, 0] + 0.5).astype(np.float32)
    return features, target
