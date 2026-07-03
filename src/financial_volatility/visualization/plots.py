"""Plot generation for benchmark result CSV files."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from matplotlib import pyplot as plt  # noqa: E402

PLOT_FILENAMES = {
    "rmse": "rmse_by_model.png",
    "mae": "mae_by_model.png",
    "training_time_seconds": "training_time_by_model.png",
    "inference_time_seconds": "inference_time_by_model.png",
    "peak_memory_mb": "memory_by_model.png",
    "accuracy_vs_cost": "accuracy_vs_compute_cost.png",
}


def generate_result_plots(
    results_csv: str | Path,
    output_dir: str | Path,
) -> tuple[Path, ...]:
    """Generate thesis-ready PNG plots from benchmark results."""
    results = pd.read_csv(results_csv)
    if results.empty:
        raise ValueError("results CSV must contain at least one row")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    generated_paths = [
        _bar_plot(
            results,
            metric_column="rmse",
            ylabel="RMSE",
            title="RMSE by Model",
            output_path=output_path / PLOT_FILENAMES["rmse"],
        ),
        _bar_plot(
            results,
            metric_column="mae",
            ylabel="MAE",
            title="MAE by Model",
            output_path=output_path / PLOT_FILENAMES["mae"],
        ),
        _bar_plot(
            results,
            metric_column="training_time_seconds",
            ylabel="Seconds",
            title="Training Time by Model",
            output_path=output_path / PLOT_FILENAMES["training_time_seconds"],
        ),
        _bar_plot(
            results,
            metric_column="inference_time_seconds",
            ylabel="Seconds",
            title="Inference Time by Model",
            output_path=output_path / PLOT_FILENAMES["inference_time_seconds"],
        ),
        _bar_plot(
            results,
            metric_column="peak_memory_mb",
            ylabel="MB",
            title="Peak Memory by Model",
            output_path=output_path / PLOT_FILENAMES["peak_memory_mb"],
        ),
        _scatter_plot(
            results,
            output_path=output_path / PLOT_FILENAMES["accuracy_vs_cost"],
        ),
    ]
    return tuple(generated_paths)


def _bar_plot(
    results: pd.DataFrame,
    *,
    metric_column: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> Path:
    """Save one metric-by-model bar plot."""
    _require_columns(results, ("model_name", metric_column))
    grouped = (
        results[["model_name", metric_column]]
        .dropna()
        .groupby("model_name", as_index=True)
        .mean(numeric_only=True)
        .sort_index()
    )
    if grouped.empty:
        raise ValueError(f"No values available for plot column: {metric_column}")

    fig, ax = plt.subplots(figsize=(6.0, 4.0), layout="constrained")
    ax.bar(grouped.index.astype(str), grouped[metric_column], color="#2f6f9f")
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _scatter_plot(results: pd.DataFrame, *, output_path: Path) -> Path:
    """Save an accuracy-vs-compute-cost scatter plot."""
    _require_columns(results, ("model_name", "rmse", "training_time_seconds"))
    plot_data = results[["model_name", "rmse", "training_time_seconds"]].dropna()
    if plot_data.empty:
        raise ValueError("No values available for accuracy-vs-cost plot")

    fig, ax = plt.subplots(figsize=(6.0, 4.0), layout="constrained")
    for model_name, group in plot_data.groupby("model_name", sort=True):
        ax.scatter(
            group["training_time_seconds"],
            group["rmse"],
            label=str(model_name),
        )
    ax.set_title("Accuracy vs Computational Cost")
    ax.set_xlabel("Training time (seconds)")
    ax.set_ylabel("RMSE")
    ax.legend()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _require_columns(results: pd.DataFrame, columns: tuple[str, ...]) -> None:
    """Validate required result columns."""
    missing_columns = [column for column in columns if column not in results.columns]
    if missing_columns:
        msg = f"results CSV is missing required columns: {missing_columns}"
        raise ValueError(msg)


__all__ = ["PLOT_FILENAMES", "generate_result_plots"]
