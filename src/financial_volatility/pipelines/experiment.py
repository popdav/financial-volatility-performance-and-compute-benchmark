"""Minimal local experiment pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from financial_volatility.benchmark import BenchmarkRunner, HardwareTarget
from financial_volatility.benchmark.types import ModelInput, PredictionContext
from financial_volatility.data.loaders import load_ohlcv_csv
from financial_volatility.data.splitting import split_time_series
from financial_volatility.evaluation.results import ExperimentResult
from financial_volatility.features.engineering import build_volatility_features
from financial_volatility.models import ForecastModel
from financial_volatility.results.storage import write_experiment_results_csv


@dataclass(frozen=True, slots=True)
class ExperimentPipeline:
    """Run a local CSV volatility forecasting experiment for one provided model."""

    csv_path: str | Path
    model: ForecastModel
    results_path: str | Path
    test_size: float | None = None
    split_date: str | date | pd.Timestamp | None = None
    date_column: str = "date"
    target_column: str = "realized_volatility_5d"
    symbol: str | None = None
    dataset_name: str = "local_csv"
    hardware_target: HardwareTarget | str = HardwareTarget.CPU

    def run(self) -> ExperimentResult:
        """Execute load, feature generation, split, benchmark, and result storage."""
        market_data = load_ohlcv_csv(
            self.csv_path,
            date_column=self.date_column,
            symbol=self.symbol,
        )
        features = build_volatility_features(market_data)
        _validate_target_column(features, self.target_column)

        split = split_time_series(
            features,
            test_size=self.test_size,
            split_date=self.split_date,
            name=self.dataset_name,
        )
        train_frame = split.train.to_dataframe()
        test_frame = split.test.to_dataframe()

        training_data = ModelInput(
            features=_feature_frame(train_frame, self.target_column),
            target=train_frame[self.target_column],
            timestamps=tuple(train_frame.index),
        )
        test_input_data = PredictionContext(
            features=_feature_frame(test_frame, self.target_column),
            timestamps=tuple(test_frame.index),
        )

        result = BenchmarkRunner(
            model=self.model,
            training_data=training_data,
            test_input_data=test_input_data,
            test_target_data=test_frame[self.target_column],
            hardware_target=self.hardware_target,
            dataset_name=self.dataset_name,
            target_name=self.target_column,
        ).run()
        write_experiment_results_csv(result, self.results_path)
        return result


def _validate_target_column(features: pd.DataFrame, target_column: str) -> None:
    """Validate that engineered features include the requested target column."""
    if target_column not in features.columns:
        msg = f"Feature data is missing target column: {target_column}"
        raise ValueError(msg)


def _feature_frame(frame: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Return model input features with the target column removed."""
    return frame.drop(columns=[target_column]).copy(deep=True)
