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
from financial_volatility.features.engineering import build_supervised_dataset
from financial_volatility.features.sequences import build_sequence_dataset
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
    target_horizon: int = 5
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
        features, target = build_supervised_dataset(
            market_data,
            horizon=self.target_horizon,
        )
        supervised = features.join(target)

        split = split_time_series(
            supervised,
            test_size=self.test_size,
            split_date=self.split_date,
            name=self.dataset_name,
        )
        train_frame = split.train.to_dataframe()
        test_frame = split.test.to_dataframe()
        target_column = str(target.name)
        training_data, test_input_data, test_target_data = _model_datasets(
            self.model,
            train_frame,
            test_frame,
            target_column,
        )

        result = BenchmarkRunner(
            model=self.model,
            training_data=training_data,
            test_input_data=test_input_data,
            test_target_data=test_target_data,
            hardware_target=self.hardware_target,
            dataset_name=self.dataset_name,
            target_name=target_column,
            forecast_horizon=self.target_horizon,
        ).run()
        write_experiment_results_csv(result, self.results_path)
        return result


def _model_datasets(
    model: ForecastModel,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    target_column: str,
) -> tuple[ModelInput, PredictionContext, object]:
    """Build tabular or sequence model inputs from split supervised frames."""
    sequence_length = getattr(model, "sequence_length", None)
    if sequence_length is None:
        train_features = _feature_frame(train_frame, target_column)
        test_features = _feature_frame(test_frame, target_column)
        return (
            ModelInput(
                features=train_features,
                target=train_frame[target_column],
                timestamps=tuple(train_frame.index),
            ),
            PredictionContext(
                features=test_features,
                timestamps=tuple(test_frame.index),
            ),
            test_frame[target_column],
        )

    sequence_length = int(sequence_length)
    train_sequences = build_sequence_dataset(
        _feature_frame(train_frame, target_column),
        train_frame[target_column],
        sequence_length=sequence_length,
    )
    test_sequences = build_sequence_dataset(
        _feature_frame(test_frame, target_column),
        test_frame[target_column],
        sequence_length=sequence_length,
    )
    return (
        ModelInput(
            features=train_sequences.X,
            target=train_sequences.y,
            timestamps=train_sequences.target_timestamps,
        ),
        PredictionContext(
            features=test_sequences.X,
            timestamps=test_sequences.target_timestamps,
        ),
        test_sequences.y,
    )


def _feature_frame(frame: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Return model input features with the target column removed."""
    return frame.drop(columns=[target_column]).copy(deep=True)
