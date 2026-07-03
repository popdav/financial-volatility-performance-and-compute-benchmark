"""LSTM forecasting model adapter."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Self, cast

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from financial_volatility.benchmark.hardware import resolve_torch_device
from financial_volatility.benchmark.types import (
    Forecast,
    HardwareTarget,
    ModelInput,
    ModelMetadata,
    PredictionContext,
)
from financial_volatility.models.base import ForecastModel


@dataclass(frozen=True, slots=True)
class LSTMConfig:
    """Configuration for the LSTM volatility forecaster."""

    sequence_length: int
    hidden_size: int = 16
    num_layers: int = 1
    dropout: float = 0.0
    epochs: int = 10
    learning_rate: float = 0.001
    device: str = "cpu"


@dataclass(slots=True)
class LSTMModel(ForecastModel):
    """PyTorch LSTM adapter for sequence-based volatility forecasting."""

    sequence_length: int
    hidden_size: int = 16
    num_layers: int = 1
    dropout: float = 0.0
    epochs: int = 10
    learning_rate: float = 0.001
    device: str = "cpu"
    _network: _LSTMRegressor | None = field(default=None, init=False, repr=False)
    _input_size: int | None = field(default=None, init=False, repr=False)
    _is_trained: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate LSTM hyperparameters."""
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")

        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")

        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")

        if self.dropout < 0.0:
            raise ValueError("dropout must be non-negative")

        if self.epochs <= 0:
            raise ValueError("epochs must be positive")

        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")

    def train(
        self,
        data: ModelInput,
        validation_data: ModelInput | None = None,
    ) -> None:
        """Fit the LSTM regressor on sequence data."""
        _ = validation_data
        features = _sequence_array(data.features, self.sequence_length)
        target = _target_tensor(data.target)

        if len(features) != len(target):
            msg = (
                "LSTMModel requires features and target with the same length: "
                f"{len(features)} != {len(target)}"
            )
            raise ValueError(msg)

        device = _resolve_device(self.device)
        self._input_size = features.shape[2]
        self._network = _LSTMRegressor(
            input_size=self._input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
        ).to(device)
        self._network.train()

        x_tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
        y_tensor = target.to(device)
        optimizer = torch.optim.Adam(self._network.parameters(), lr=self.learning_rate)
        loss_function = nn.MSELoss()

        for _epoch in range(self.epochs):
            optimizer.zero_grad()
            prediction = self._network(x_tensor)
            loss = loss_function(prediction, y_tensor)
            loss.backward()
            optimizer.step()

        self._network.eval()
        self._is_trained = True

    def predict(self, context: PredictionContext, horizon: int) -> Forecast:
        """Predict realized volatility for the provided sequence rows."""
        if self._network is None or not self._is_trained:
            raise ValueError("LSTMModel must be trained before prediction")

        if horizon <= 0:
            raise ValueError("horizon must be positive")

        features = _sequence_array(context.features, self.sequence_length)
        if len(features) < horizon:
            msg = (
                "Prediction context does not contain enough sequence rows for "
                f"horizon {horizon}: {len(features)} rows"
            )
            raise ValueError(msg)

        device = _resolve_device(self.device)
        self._network.to(device)
        self._network.eval()
        with torch.no_grad():
            x_tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
            predictions = self._network(x_tensor).detach().cpu().numpy()

        values = np.asarray(predictions[:horizon], dtype=np.float64)
        return Forecast(
            values=values.tolist(),
            horizon=horizon,
            timestamps=context.timestamps,
        )

    def save(self, path: str | Path) -> None:
        """Persist model configuration and trained weights."""
        if self._network is None or self._input_size is None:
            raise ValueError("LSTMModel must be trained before saving")

        checkpoint = {
            "config": asdict(self._config()),
            "input_size": self._input_size,
            "state_dict": self._network.cpu().state_dict(),
            "is_trained": self._is_trained,
        }
        torch.save(checkpoint, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load a previously persisted LSTM adapter."""
        checkpoint = torch.load(Path(path), map_location="cpu")
        if not isinstance(checkpoint, dict):
            raise TypeError("Persisted object is not an LSTM checkpoint")

        config = LSTMConfig(**cast(dict[str, Any], checkpoint["config"]))
        model = cls(**asdict(config))
        input_size = int(checkpoint["input_size"])
        model._input_size = input_size
        model._network = _LSTMRegressor(
            input_size=input_size,
            hidden_size=model.hidden_size,
            num_layers=model.num_layers,
            dropout=model.dropout if model.num_layers > 1 else 0.0,
        )
        model._network.load_state_dict(checkpoint["state_dict"])
        model._network.eval()
        model._is_trained = bool(checkpoint["is_trained"])
        return model

    def metadata(self) -> ModelMetadata:
        """Return reproducibility metadata for benchmark result records."""
        return ModelMetadata(
            name="lstm",
            model_family="deep_learning",
            parameters=asdict(self._config()),
            supported_hardware=(HardwareTarget.CPU, HardwareTarget.CUDA),
        )

    def _config(self) -> LSTMConfig:
        """Return the public configuration as a dataclass."""
        return LSTMConfig(
            sequence_length=self.sequence_length,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            device=self.device,
        )


class _LSTMRegressor(nn.Module):
    """Small sequence-to-one LSTM regressor."""

    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return one prediction per input sequence."""
        output, _state = self.lstm(features)
        prediction = self.output(output[:, -1, :]).squeeze(-1)
        return cast(torch.Tensor, prediction)


def _sequence_array(
    features: object,
    sequence_length: int,
) -> npt.NDArray[np.float32]:
    """Convert sequence features to a finite three-dimensional array."""
    values = np.asarray(cast(npt.ArrayLike, features), dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(
            "LSTMModel features must have shape (samples, sequence, features)"
        )

    if values.shape[0] == 0:
        raise ValueError("LSTMModel features must be non-empty")

    if values.shape[1] != sequence_length:
        msg = (
            "LSTMModel feature sequence length does not match configuration: "
            f"{values.shape[1]} != {sequence_length}"
        )
        raise ValueError(msg)

    if values.shape[2] == 0:
        raise ValueError("LSTMModel features must include at least one feature")

    if not np.all(np.isfinite(values)):
        raise ValueError("LSTMModel features must be finite")

    return values


def _target_tensor(target: object) -> torch.Tensor:
    """Convert training targets to a finite one-dimensional tensor."""
    values = np.asarray(cast(npt.ArrayLike, target), dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError("LSTMModel target must be non-empty")

    if not np.all(np.isfinite(values)):
        raise ValueError("LSTMModel target must be finite")

    return torch.as_tensor(values, dtype=torch.float32)


def _resolve_device(device: str) -> torch.device:
    """Resolve a configured torch device, gracefully handling auto mode."""
    return torch.device(resolve_torch_device(device))


__all__ = ["LSTMConfig", "LSTMModel"]
