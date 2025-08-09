"""Tuning package for hyperparameter optimization (HPO).

Modules:
- purged_tscv: Purged/embargoed time series CV splitter to mitigate leakage
- search_spaces: Model-specific parameter spaces and Optuna suggest helpers
- optuna_tuner: End-to-end Optuna-based tuner that reads/writes to Postgres
"""
