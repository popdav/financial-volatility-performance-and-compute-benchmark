"""Deterministic thesis table generation from benchmark results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

SUMMARY_COLUMNS = (
    "model_name",
    "model_type",
    "hardware",
    "rmse",
    "mae",
    "mape",
    "training_time_seconds",
    "inference_time_seconds",
    "peak_memory_mb",
    "model_size_mb",
)


@dataclass(frozen=True, slots=True)
class ThesisTablePaths:
    """Generated thesis table paths."""

    markdown: Path
    csv: Path
    latex: Path | None = None


def generate_thesis_tables(
    results_csv: str | Path,
    output_dir: str | Path,
    *,
    include_latex: bool = False,
) -> ThesisTablePaths:
    """Generate deterministic Markdown, CSV, and optional LaTeX summary tables."""
    results = pd.read_csv(results_csv)
    if results.empty:
        raise ValueError("results CSV must contain at least one row")

    summary = summarize_results(results)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_path = output_path / "thesis_summary.csv"
    markdown_path = output_path / "thesis_summary.md"
    latex_path = output_path / "thesis_summary.tex" if include_latex else None

    summary.to_csv(csv_path, index=False, float_format="%.6g")
    markdown_path.write_text(_to_markdown(summary), encoding="utf-8")
    if latex_path is not None:
        latex_path.write_text(summary.to_latex(index=False), encoding="utf-8")

    return ThesisTablePaths(markdown=markdown_path, csv=csv_path, latex=latex_path)


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate benchmark results by model, type, and hardware."""
    _require_columns(results, SUMMARY_COLUMNS)
    summary = (
        results.loc[:, list(SUMMARY_COLUMNS)]
        .groupby(["model_name", "model_type", "hardware"], as_index=False)
        .mean(numeric_only=True)
        .sort_values(["model_name", "hardware"], kind="stable")
        .reset_index(drop=True)
    )
    return summary.loc[:, list(SUMMARY_COLUMNS)]


def _require_columns(results: pd.DataFrame, columns: tuple[str, ...]) -> None:
    """Validate required result columns."""
    missing_columns = [column for column in columns if column not in results.columns]
    if missing_columns:
        msg = f"results CSV is missing required columns: {missing_columns}"
        raise ValueError(msg)


def _to_markdown(summary: pd.DataFrame) -> str:
    """Render a simple GitHub-flavored Markdown table without extra dependencies."""
    columns = [str(column) for column in summary.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _column in columns) + " |",
    ]
    for row in summary.itertuples(index=False, name=None):
        values = [_format_table_value(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines) + "\n"


def _format_table_value(value: object) -> str:
    """Format table cells deterministically."""
    if isinstance(value, float):
        return f"{value:.6g}"

    return str(value)


__all__ = ["ThesisTablePaths", "generate_thesis_tables", "summarize_results"]
