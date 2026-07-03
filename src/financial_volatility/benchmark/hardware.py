"""Hardware detection helpers for benchmark runs."""

from __future__ import annotations

import torch

from financial_volatility.benchmark.types import HardwareTarget


def is_cuda_available() -> bool:
    """Return whether PyTorch can run CUDA workloads in this environment."""
    return torch.cuda.is_available()


def available_hardware_targets() -> tuple[HardwareTarget, ...]:
    """Return benchmark hardware targets available on this machine."""
    targets = [HardwareTarget.CPU]
    if is_cuda_available():
        targets.append(HardwareTarget.CUDA)

    return tuple(targets)


def resolve_torch_device(device: str) -> str:
    """Resolve auto/cuda/cpu device strings for PyTorch-backed models."""
    if device == "auto":
        return "cuda" if is_cuda_available() else "cpu"

    if device == "cuda" and not is_cuda_available():
        raise ValueError("CUDA device requested but torch.cuda is not available")

    return device


__all__ = ["available_hardware_targets", "is_cuda_available", "resolve_torch_device"]
