"""Typer command-line interface for benchmark workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer
from pydantic import ValidationError

from financial_volatility.benchmark import (
    HardwareTarget,
    ScalarValue,
)
from financial_volatility.config import (
    BenchmarkSettings,
    DatasetSettings,
    load_settings,
)
from financial_volatility.data.preparation import prepare_dataset
from financial_volatility.models import ForecastModel, ModelRegistry
from financial_volatility.pipelines.experiment import ExperimentPipeline

app = typer.Typer(help="Financial volatility benchmark CLI.")
dataset_app = typer.Typer(help="Acquire and inspect research datasets.")
app.add_typer(dataset_app, name="dataset")
VALIDATE_CONFIG_OPTION = typer.Option(
    ...,
    "--config",
    "-c",
    help="Path to YAML config.",
)
RUN_CONFIG_OPTION = typer.Option(
    ...,
    "--config",
    "-c",
    help="Path to YAML config.",
)
DATASET_CONFIG_OPTION = typer.Option(..., "--config", "-c", help="Dataset YAML config.")
FORCE_REFRESH_OPTION = typer.Option(False, "--force-refresh")


@dataset_app.command("prepare")
def dataset_prepare(
    config: Path = DATASET_CONFIG_OPTION,
    force_refresh: bool = FORCE_REFRESH_OPTION,
) -> None:
    """Materialize a normalized and validated research dataset."""
    try:
        result = prepare_dataset(config, force_refresh=force_refresh)
    except (ValueError, ValidationError) as error:
        typer.echo(f"Dataset preparation failed: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(f"Dataset: {result.dataset_path}")
    typer.echo(f"Metadata: {result.metadata_path}")
    typer.echo(f"Report: {result.report_path}")


@app.command("validate-config")
def validate_config(
    config: Path = VALIDATE_CONFIG_OPTION,
) -> None:
    """Validate a benchmark configuration file."""
    settings = _load_settings_for_cli(config)
    typer.echo(f"Valid config: {config}")
    typer.echo(f"Model: {settings.model.name or 'unset'}")


@app.command("list-models")
def list_models() -> None:
    """List registered model names."""
    for model_name in ModelRegistry.names():
        typer.echo(model_name)


@app.command("run")
def run(
    config: Path = RUN_CONFIG_OPTION,
) -> None:
    """Run a configuration-driven benchmark experiment."""
    settings = _load_settings_for_cli(config)
    output_dir = settings.output.directory
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = _resolve_csv_path(settings.dataset, output_dir)
    result_path = output_dir / "results.csv"
    pipeline = ExperimentPipeline(
        csv_path=csv_path,
        model=_create_model(settings.model.name, settings.model.parameters),
        results_path=result_path,
        test_size=float(getattr(settings.dataset, "test_size", 0.2)),
        date_column=str(getattr(settings.dataset, "date_column", "date")),
        target_horizon=settings.target.horizon,
        dataset_name=str(getattr(settings.dataset, "name", settings.dataset.provider)),
        hardware_target=HardwareTarget(settings.hardware.device),
    )
    try:
        result = pipeline.run()
    except ValueError as error:
        typer.echo(f"Experiment failed: {error}", err=True)
        raise typer.Exit(1) from error

    typer.echo(f"Experiment complete: {result.experiment_id}")
    typer.echo(f"Results: {result_path}")


def _load_settings_for_cli(config: Path) -> BenchmarkSettings:
    """Load settings and convert expected failures into concise CLI errors."""
    try:
        return load_settings(config)
    except FileNotFoundError as error:
        raise typer.BadParameter(f"Config file not found: {config}") from error
    except (ValueError, ValidationError) as error:
        raise typer.BadParameter(f"Invalid config {config}: {error}") from error


def _create_model(
    model_name: str | None,
    parameters: dict[str, Any],
) -> ForecastModel:
    """Create a CLI model from settings."""
    if model_name is None:
        raise typer.BadParameter("model.name must be set")

    return ModelRegistry.create(
        model_name,
        parameters={key: _scalar_parameter(value) for key, value in parameters.items()},
    )


def _resolve_csv_path(dataset_settings: DatasetSettings, output_dir: Path) -> Path:
    """Return a local CSV path, generating synthetic data when configured."""
    path = getattr(dataset_settings, "path", None)
    if path is not None:
        return Path(str(path))

    synthetic_path = output_dir / "synthetic_ohlcv.csv"
    _write_synthetic_ohlcv_csv(synthetic_path)
    return synthetic_path


def _write_synthetic_ohlcv_csv(path: Path) -> None:
    """Create a deterministic OHLCV CSV for CLI smoke runs."""
    dates = pd.date_range("2026-01-01", periods=80, freq="D")
    close = np.linspace(100.0, 125.0, num=len(dates))
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.linspace(1000.0, 1500.0, num=len(dates)),
        }
    )
    frame.to_csv(path, index=False)


def _scalar_parameter(value: object) -> ScalarValue:
    """Validate CLI model parameters supported by the registry."""
    if isinstance(value, str | int | float | bool) or value is None:
        return value

    msg = f"Model parameters must be scalar values, got {type(value).__name__}"
    raise typer.BadParameter(msg)


def main() -> None:
    """CLI entry point."""
    app()


__all__ = ["app", "main"]
