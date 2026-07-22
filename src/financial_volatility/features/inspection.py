"""Inspection reports and plots for engineered volatility features."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

from financial_volatility.features.engineering import FeatureMetadata

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def generate_feature_inspection(
    features: pd.DataFrame,
    metadata: tuple[FeatureMetadata, ...],
    output_directory: str | Path = "results/features",
) -> tuple[Path, ...]:
    """Write tabular summaries and the required feature inspection plots."""
    output = Path(output_directory)
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    summary_path = output / "feature_summary.csv"
    correlation_path = output / "correlation_matrix.csv"
    distribution_path = output / "feature_distributions.csv"
    missing_path = output / "feature_missing_values.csv"
    pd.DataFrame(
        [
            {
                "feature": item.name,
                "category": item.category,
                "rolling_window": item.rolling_window,
                "description": item.description,
            }
            for item in metadata
        ]
    ).to_csv(summary_path, index=False)
    features.corr().to_csv(correlation_path)
    features.describe().T.to_csv(distribution_path)
    features.isna().sum().rename("missing_values").to_csv(missing_path)

    heatmap_path = plots / "correlation_heatmap.png"
    fig, ax = plt.subplots(figsize=(10, 8), layout="constrained")
    image = ax.imshow(features.corr(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(features.columns)), features.columns, rotation=90)
    ax.set_yticks(range(len(features.columns)), features.columns)
    fig.colorbar(image, ax=ax, label="Pearson correlation")
    ax.set_title("Feature Correlation Heatmap")
    fig.savefig(heatmap_path, dpi=150)
    plt.close(fig)

    generated = [
        summary_path,
        correlation_path,
        distribution_path,
        missing_path,
        heatmap_path,
    ]
    generated.append(
        _line_plot(
            features,
            "historical_volatility_",
            "Rolling Volatility",
            plots / "rolling_volatility.png",
        )
    )
    generated.append(_line_plot(features, "rsi_", "RSI", plots / "rsi.png"))
    generated.append(
        _line_plot(features, "atr_", "Average True Range", plots / "atr.png")
    )

    histogram_path = plots / "feature_histograms.png"
    axes = features.hist(figsize=(14, 10), bins=30)
    fig = axes.ravel()[0].figure
    fig.suptitle("Feature Distributions")
    fig.tight_layout()
    fig.savefig(histogram_path, dpi=150)
    plt.close(fig)
    generated.append(histogram_path)
    return tuple(generated)


def _line_plot(
    features: pd.DataFrame, prefix: str, title: str, output_path: Path
) -> Path:
    columns = [str(column) for column in features if str(column).startswith(prefix)]
    if not columns:
        raise ValueError(f"inspection requires at least one {prefix!r} feature")
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    features[columns].plot(ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Date")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


__all__ = ["generate_feature_inspection"]
