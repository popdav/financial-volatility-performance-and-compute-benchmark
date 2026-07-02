"""Configuration settings tests."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from financial_volatility.config.settings import BenchmarkSettings, load_settings


def test_valid_yaml_config_loads_successfully(tmp_path: Path) -> None:
    """A complete generic YAML config validates into benchmark settings."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
dataset:
  provider: yahoo_finance
  symbols: []
  start_date: null
  end_date: null
target:
  name: realized_volatility
  horizon: 1
model:
  name: null
  parameters: {}
hardware:
  device: cpu
output:
  directory: results/
""",
        encoding="utf-8",
    )

    settings = load_settings(config_path)

    assert isinstance(settings, BenchmarkSettings)
    assert settings.dataset.provider == "yahoo_finance"
    assert settings.dataset.symbols == []
    assert settings.target.name == "realized_volatility"
    assert settings.target.horizon == 1
    assert settings.model.name is None
    assert settings.model.parameters == {}
    assert settings.hardware.device == "cpu"
    assert settings.output.directory == Path("results")


def test_invalid_horizon_values_fail_validation(tmp_path: Path) -> None:
    """Target horizon must be a positive integer."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(_config_yaml(horizon=0), encoding="utf-8")

    with pytest.raises(ValidationError, match="horizon"):
        load_settings(config_path)


def test_invalid_hardware_device_values_fail_validation(tmp_path: Path) -> None:
    """Hardware device values are restricted to known execution targets."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(_config_yaml(device="quantum"), encoding="utf-8")

    with pytest.raises(ValidationError, match="device"):
        load_settings(config_path)


def test_empty_symbols_list_is_allowed(tmp_path: Path) -> None:
    """The config foundation does not require concrete tickers yet."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(_config_yaml(symbols="[]"), encoding="utf-8")

    settings = load_settings(config_path)

    assert settings.dataset.symbols == []


def _config_yaml(
    *,
    horizon: int = 1,
    device: str = "cpu",
    symbols: str = "[]",
) -> str:
    """Build a minimal benchmark settings YAML document."""
    return f"""
dataset:
  provider: yahoo_finance
  symbols: {symbols}
  start_date: null
  end_date: null
target:
  name: realized_volatility
  horizon: {horizon}
model:
  name: null
  parameters: {{}}
hardware:
  device: {device}
output:
  directory: results/
"""
