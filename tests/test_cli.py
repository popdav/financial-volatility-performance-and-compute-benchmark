"""CLI tests."""

from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from financial_volatility.cli import app


def test_cli_help_works() -> None:
    """The CLI exposes help text."""
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "run" in result.output
    assert "validate-config" in result.output
    assert "list-models" in result.output


def test_cli_validate_config_works(tmp_path: Path) -> None:
    """Config validation loads a valid YAML config."""
    config_path = _config(tmp_path)

    result = CliRunner().invoke(app, ["validate-config", "--config", str(config_path)])

    assert result.exit_code == 0
    assert "Valid config" in result.output


def test_cli_list_models_works() -> None:
    """Registered models are listed."""
    result = CliRunner().invoke(app, ["list-models"])

    assert result.exit_code == 0
    assert "xgboost" in result.output
    assert "linear_regression" in result.output
    assert "dummy" not in result.output


def test_cli_synthetic_experiment_can_run_with_registered_model(tmp_path: Path) -> None:
    """The CLI can run a synthetic experiment with a real registered model."""
    config_path = _config(tmp_path, model_name="linear_regression")

    result = CliRunner().invoke(app, ["run", "--config", str(config_path)])

    results_path = tmp_path / "results.csv"
    rows = pd.read_csv(results_path)
    assert result.exit_code == 0
    assert "Experiment complete" in result.output
    assert results_path.exists()
    assert rows.loc[0, "model_name"] == "linear_regression"


@pytest.mark.parametrize(
    ("model_name", "parameters"),
    [
        ("garch", "{}"),
        ("linear_regression", "{}"),
        ("xgboost", "{n_estimators: 4, max_depth: 2, learning_rate: 0.2}"),
    ],
)
def test_cli_real_experiment_can_run(
    tmp_path: Path,
    model_name: str,
    parameters: str,
) -> None:
    """The CLI can run registered real models from config."""
    config_path = _config(tmp_path, model_name=model_name, parameters=parameters)

    result = CliRunner().invoke(app, ["run", "--config", str(config_path)])

    results_path = tmp_path / "results.csv"
    rows = pd.read_csv(results_path)
    assert result.exit_code == 0
    assert "Experiment complete" in result.output
    assert rows.loc[0, "model_name"] == model_name


def _config(
    tmp_path: Path,
    *,
    model_name: str = "linear_regression",
    parameters: str = "{}",
) -> Path:
    """Write a minimal CLI config file."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
dataset:
  provider: synthetic
  symbols: []
  path: null
  test_size: 0.2
target:
  name: realized_volatility
  horizon: 5
model:
  name: {model_name}
  parameters: {parameters}
hardware:
  device: cpu
output:
  directory: {tmp_path}
""",
        encoding="utf-8",
    )
    return config_path
