"""Typer command-line interface for benchmark workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import typer

from financial_volatility.benchmark import (
    Forecast,
    HardwareTarget,
    ModelInput,
    ModelMetadata,
    PredictionContext,
    ScalarValue,
)
from financial_volatility.config import (
    BenchmarkSettings,
    DatasetSettings,
    load_settings,
)
from financial_volatility.models import ForecastModel, ModelRegistry
from financial_volatility.pipelines.experiment import ExperimentPipeline

app = typer.Typer(help="Financial volatility benchmark CLI.")
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


@app.command("validate-config")
def validate_config(
    config: Path = VALIDATE_CONFIG_OPTION,
) -> None:
    """Validate a benchmark configuration file."""
    settings = load_settings(config)
    typer.echo(f"Valid config: {config}")
    typer.echo(f"Model: {settings.model.name or 'unset'}")


@app.command("list-models")
def list_models() -> None:
    """List registered model names."""
    for model_name in ModelRegistry.names():
        typer.echo(model_name)
    typer.echo("dummy")


@app.command("run")
def run(
    config: Path = RUN_CONFIG_OPTION,
) -> None:
    """Run a configuration-driven benchmark experiment."""
    settings = load_settings(config)
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
        target_column=_target_column(settings),
        dataset_name=str(getattr(settings.dataset, "name", settings.dataset.provider)),
        hardware_target=HardwareTarget(settings.hardware.device),
    )
    result = pipeline.run()
    typer.echo(f"Experiment complete: {result.experiment_id}")
    typer.echo(f"Results: {result_path}")


def _create_model(
    model_name: str | None,
    parameters: dict[str, Any],
) -> ForecastModel:
    """Create a CLI model from settings."""
    if model_name == "dummy":
        return DummyForecastModel()

    if model_name is None:
        raise typer.BadParameter("model.name must be set")

    return ModelRegistry.create(
        model_name,
        parameters={key: _scalar_parameter(value) for key, value in parameters.items()},
    )


class DummyForecastModel(ForecastModel):
    """Simple mean forecast model for CLI smoke tests."""

    def __init__(self) -> None:
        """Create an untrained dummy model."""
        self._mean_target: float | None = None

    def train(
        self,
        data: ModelInput,
        validation_data: ModelInput | None = None,
    ) -> None:
        """Store the mean target value."""
        _ = validation_data
        target = np.asarray(cast(npt.ArrayLike, data.target), dtype=np.float64)
        if target.size == 0:
            raise ValueError("DummyForecastModel target must be non-empty")

        self._mean_target = float(np.mean(target))

    def predict(self, context: PredictionContext, horizon: int) -> Forecast:
        """Return the training target mean for each requested horizon step."""
        _ = context
        if self._mean_target is None:
            raise ValueError("DummyForecastModel must be trained before prediction")

        return Forecast(values=[self._mean_target] * horizon, horizon=horizon)

    def save(self, path: str | Path) -> None:
        """Persist the dummy scalar state."""
        Path(path).write_text(str(self._mean_target), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load a previously persisted dummy model."""
        model = cls()
        model._mean_target = float(Path(path).read_text(encoding="utf-8"))
        return model

    def metadata(self) -> ModelMetadata:
        """Return reproducibility metadata for benchmark result records."""
        return ModelMetadata(
            name="dummy",
            model_family="test",
            supported_hardware=(HardwareTarget.CPU,),
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


def _target_column(settings: BenchmarkSettings) -> str:
    """Resolve the target column expected after feature engineering."""
    target_settings = settings.target
    explicit_column = getattr(target_settings, "column", None)
    if explicit_column is not None:
        return str(explicit_column)

    target_name = target_settings.name
    horizon = target_settings.horizon
    if target_name == "realized_volatility":
        return f"realized_volatility_{horizon}d"

    return target_name


def _scalar_parameter(value: object) -> ScalarValue:
    """Validate CLI model parameters supported by the registry."""
    if isinstance(value, str | int | float | bool) or value is None:
        return value

    msg = f"Model parameters must be scalar values, got {type(value).__name__}"
    raise typer.BadParameter(msg)


def main() -> None:
    """CLI entry point."""
    app()


__all__ = ["DummyForecastModel", "app", "main"]
