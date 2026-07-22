"""Configuration-driven experiment pipeline tests."""

from pathlib import Path
from typing import Self, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from financial_volatility.benchmark import (
    Forecast,
    HardwareTarget,
    ModelInput,
    ModelMetadata,
    PredictionContext,
)
from financial_volatility.models import ForecastModel
from financial_volatility.pipelines.configured import run_experiment_from_config


class DummyConfigModel(ForecastModel):
    """Small model used for config-driven pipeline tests."""

    def __init__(self) -> None:
        """Create an untrained dummy model."""
        self.mean_target: float | None = None

    def train(
        self,
        data: ModelInput,
        validation_data: ModelInput | None = None,
    ) -> None:
        """Store the mean target."""
        _ = validation_data
        target = np.asarray(cast(npt.ArrayLike, data.target), dtype=np.float64)
        self.mean_target = float(np.mean(target))

    def predict(self, context: PredictionContext, horizon: int) -> Forecast:
        """Return mean forecasts."""
        _ = context
        assert self.mean_target is not None
        return Forecast(values=[self.mean_target] * horizon, horizon=horizon)

    def save(self, path: str | Path) -> None:
        """No-op save."""
        _ = path

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load a dummy model."""
        _ = path
        return cls()

    def metadata(self) -> ModelMetadata:
        """Return stable metadata."""
        return ModelMetadata(
            name="dummy",
            model_family="test",
            supported_hardware=(HardwareTarget.CPU,),
        )


def test_config_driven_dummy_experiment_works(tmp_path: Path) -> None:
    """A config file can drive an experiment with an injected dummy model."""
    csv_path = tmp_path / "ohlcv.csv"
    _ohlcv_frame().to_csv(csv_path, index_label="date")
    config_path = _config(tmp_path, csv_path)

    result = run_experiment_from_config(
        config_path,
        model_overrides={"dummy": DummyConfigModel()},
    )

    assert result.model.name == "dummy"
    assert result.target_name == "future_realized_volatility_5d"
    assert result.horizon == 5
    assert (tmp_path / "results.csv").exists()


def test_invalid_config_provider_fails_clearly(tmp_path: Path) -> None:
    """Unsupported dataset providers fail clearly."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
dataset:
  provider: yahoo_finance
  symbols: []
target:
  name: realized_volatility
  horizon: 5
model:
  name: dummy
  parameters: {{}}
hardware:
  device: cpu
output:
  directory: {tmp_path}
""",
        encoding="utf-8",
    )

    try:
        run_experiment_from_config(
            config_path,
            model_overrides={"dummy": DummyConfigModel()},
        )
    except ValueError as error:
        assert "Unsupported config dataset provider" in str(error)
    else:
        raise AssertionError("Expected unsupported provider to fail")


def _config(tmp_path: Path, csv_path: Path) -> Path:
    """Write a valid local CSV experiment config."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
dataset:
  provider: local_csv
  symbols: []
  path: {csv_path}
  test_size: 0.2
target:
  name: realized_volatility
  horizon: 5
model:
  name: dummy
  parameters: {{}}
hardware:
  device: cpu
output:
  directory: {tmp_path}
""",
        encoding="utf-8",
    )
    return config_path


def _ohlcv_frame() -> pd.DataFrame:
    """Create synthetic OHLCV data."""
    index = pd.date_range("2026-01-01", periods=120, freq="D")
    positions = np.arange(len(index), dtype=np.float64)
    close = 100.0 + positions / 5.0 + 2.0 * np.sin(positions)
    spread = 1.0 + positions / 100.0
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "adjusted_close": close,
            "volume": np.linspace(1000.0, 1500.0, num=len(index)),
        },
        index=index,
    )
