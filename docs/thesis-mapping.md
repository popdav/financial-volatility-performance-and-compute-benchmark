# Thesis Mapping

This document maps implementation modules to thesis chapters.

| Module | Thesis Area |
| --- | --- |
| `src/financial_volatility/data/types.py` | Data representation |
| `src/financial_volatility/data/loaders.py` | Data acquisition |
| `src/financial_volatility/data/splitting.py` | Experimental setup |
| `src/financial_volatility/data/walk_forward.py` | Walk-forward methodology |
| `src/financial_volatility/features/engineering.py` | Feature engineering |
| `src/financial_volatility/features/sequences.py` | Deep learning data preparation |
| `src/financial_volatility/models/garch.py` | Statistical models |
| `src/financial_volatility/models/linear.py` | Baseline models |
| `src/financial_volatility/models/xgboost.py` | Machine learning models |
| `src/financial_volatility/models/lstm.py` | Deep learning models |
| `src/financial_volatility/models/transformer.py` | Deep learning models |
| `src/financial_volatility/models/registry.py` | Experiment configuration |
| `src/financial_volatility/benchmark/runner.py` | Benchmark execution |
| `src/financial_volatility/benchmark/walk_forward.py` | Walk-forward evaluation |
| `src/financial_volatility/benchmark/profiler.py` | Computational efficiency |
| `src/financial_volatility/evaluation/metrics.py` | Evaluation metrics |
| `src/financial_volatility/results/storage.py` | Result persistence |
| `src/financial_volatility/visualization/plots.py` | Results visualization |
| `src/financial_volatility/visualization/tables.py` | Thesis result tables |
| `src/financial_volatility/cli.py` | Reproducible experiment execution |
