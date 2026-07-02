"""Configuration loading tests."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from volatility_benchmark.config import BenchmarkConfig, load_config


def test_load_default_config() -> None:
    """The default YAML config loads into the top-level schema."""
    config = load_config(Path("configs/default.yaml"))

    assert isinstance(config, BenchmarkConfig)
    assert config.dataset.ticker == "SPY"
    assert config.target.name == "realized_volatility"
    assert config.target.horizon == 1
    assert config.model.name == "linear_regression"
    assert config.model.parameters == {}
    assert config.hardware.device == "cpu"
    assert config.output.directory == Path("results")


def test_load_config_preserves_future_extension_fields(tmp_path: Path) -> None:
    """Unknown fields are retained so experiments can grow incrementally."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
dataset:
  ticker: spy
  start_date: "2015-01-01"
  end_date: "2025-12-31"
  frequency: daily
target:
  name: realized_volatility
  horizon: 1
model:
  name: linear_regression
  parameters:
    fit_intercept: true
hardware:
  device: cpu
output:
  directory: results/
random_seed: 42
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.dataset.ticker == "SPY"
    assert config.dataset.frequency == "daily"
    assert config.random_seed == 42


def test_load_config_rejects_invalid_horizon(tmp_path: Path) -> None:
    """Target horizon must be positive."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
dataset:
  ticker: SPY
  start_date: "2015-01-01"
  end_date: "2025-12-31"
target:
  name: realized_volatility
  horizon: 0
model:
  name: linear_regression
  parameters: {}
hardware:
  device: cpu
output:
  directory: results/
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="horizon"):
        load_config(config_path)


def test_load_config_rejects_reversed_date_range(tmp_path: Path) -> None:
    """Dataset end date cannot come before start date."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
dataset:
  ticker: SPY
  start_date: "2025-12-31"
  end_date: "2015-01-01"
target:
  name: realized_volatility
  horizon: 1
model:
  name: linear_regression
  parameters: {}
hardware:
  device: cpu
output:
  directory: results/
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="end_date"):
        load_config(config_path)


def test_load_config_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    """Top-level YAML must be an object."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="YAML mapping"):
        load_config(config_path)
