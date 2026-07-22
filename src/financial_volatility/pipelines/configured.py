"""Configuration-driven experiment orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from financial_volatility.benchmark import HardwareTarget, ScalarValue
from financial_volatility.config import BenchmarkSettings, load_settings
from financial_volatility.evaluation.results import ExperimentResult
from financial_volatility.models import ForecastModel, ModelRegistry
from financial_volatility.pipelines.experiment import ExperimentPipeline


def run_experiment_from_config(
    config_path: str | Path,
    *,
    model_overrides: Mapping[str, ForecastModel] | None = None,
) -> ExperimentResult:
    """Load config, instantiate the configured model, and run an experiment."""
    settings = load_settings(config_path)
    model = _create_model(settings, model_overrides or {})
    dataset_path = _dataset_path(settings)
    output_dir = settings.output.directory
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = ExperimentPipeline(
        csv_path=dataset_path,
        model=model,
        results_path=output_dir / "results.csv",
        test_size=float(getattr(settings.dataset, "test_size", 0.2)),
        date_column=str(getattr(settings.dataset, "date_column", "date")),
        target_horizon=settings.target.horizon,
        dataset_name=str(getattr(settings.dataset, "name", settings.dataset.provider)),
        hardware_target=HardwareTarget(settings.hardware.device),
        feature_config={"features": settings.features},
    )
    return pipeline.run()


def _create_model(
    settings: BenchmarkSettings,
    model_overrides: Mapping[str, ForecastModel],
) -> ForecastModel:
    """Create a configured model or use a supplied override."""
    model_name = settings.model.name
    if model_name is None:
        raise ValueError("model.name must be set")

    if model_name in model_overrides:
        return model_overrides[model_name]

    return ModelRegistry.create(
        model_name,
        parameters={
            key: _scalar_parameter(value)
            for key, value in settings.model.parameters.items()
        },
    )


def _dataset_path(settings: BenchmarkSettings) -> Path:
    """Resolve a local CSV dataset path from settings."""
    provider = settings.dataset.provider
    if provider != "local_csv":
        raise ValueError(f"Unsupported config dataset provider: {provider}")

    path = getattr(settings.dataset, "path", None)
    if path is None:
        raise ValueError("local_csv dataset config requires dataset.path")

    return Path(str(path))


def _scalar_parameter(value: Any) -> ScalarValue:
    """Validate model parameters supported by the registry."""
    if isinstance(value, str | int | float | bool) or value is None:
        return value

    msg = f"Model parameters must be scalar values, got {type(value).__name__}"
    raise ValueError(msg)


__all__ = ["run_experiment_from_config"]
