"""Search spaces and helpers for model hyperparameter optimization.

Provides:
- sklearn-style param distributions/grids for RandomizedSearchCV/GridSearchCV
- Optuna suggest_* wrappers to generate params per trial
- Target-aware adjustments based on target_pct (optional)

Models supported: rf, xgb, lgb, gb
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import math


def _rf_space(target_pct: float) -> Dict[str, Any]:
    # Adjust depth/leaf slightly by target pct
    if target_pct <= 0.01:
        max_depth = [3, 5, 7, 10]
        min_leaf = [2, 4, 8, 12]
    elif target_pct <= 0.02:
        max_depth = [5, 7, 10, 15]
        min_leaf = [1, 2, 4, 8]
    else:
        max_depth = [7, 10, 15, 20, None]
        min_leaf = [1, 2, 4, 8]

    return {
        "clf__n_estimators": [100, 200, 300, 500, 800],
        "clf__max_depth": max_depth,
        "clf__min_samples_split": [2, 5, 8, 10, 20],
        "clf__min_samples_leaf": min_leaf,
        "clf__max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
        "clf__bootstrap": [True, False],
        "clf__class_weight": ["balanced", "balanced_subsample", None],
        "clf__random_state": [42],
    }


def _xgb_space(_: float) -> Dict[str, Any]:
    return {
        "clf__n_estimators": [200, 400, 600, 800],
        "clf__max_depth": [3, 4, 5, 6, 7, 8, 10],
        "clf__learning_rate": [0.005, 0.01, 0.05, 0.1, 0.2, 0.3],
        "clf__subsample": [0.5, 0.6, 0.7, 0.8, 1.0],
        "clf__colsample_bytree": [0.5, 0.6, 0.7, 0.8, 1.0],
        "clf__gamma": [0.0, 0.1, 0.5, 1.0, 2.0],
        "clf__reg_alpha": [0.0, 0.1, 1.0, 5.0, 10.0],
        "clf__reg_lambda": [0.0, 0.1, 1.0, 5.0, 10.0],
        "clf__scale_pos_weight": [1.0, 2.0, 3.0, 4.0, 5.0],
        "clf__random_state": [42],
    }


def _lgb_space(_: float) -> Dict[str, Any]:
    return {
        "clf__n_estimators": [300, 600, 1000, 1500, 2000],
        "clf__max_depth": [-1, 3, 5, 7, 10, 12],
        "clf__learning_rate": [0.005, 0.01, 0.05, 0.1, 0.2],
        "clf__num_leaves": [20, 31, 63, 127, 255],
        "clf__subsample": [0.5, 0.6, 0.7, 0.8, 1.0],
        "clf__colsample_bytree": [0.5, 0.6, 0.7, 0.8, 1.0],
        "clf__reg_alpha": [0.0, 0.1, 1.0, 5.0, 10.0],
        "clf__reg_lambda": [0.0, 0.1, 1.0, 5.0, 10.0],
        "clf__min_child_samples": [5, 10, 20, 30, 50],
        "clf__is_unbalance": [True, False],
        "clf__random_state": [42],
        "clf__verbosity": [-1],
    }


def _gb_space(_: float) -> Dict[str, Any]:
    return {
        "clf__n_estimators": [100, 200, 300, 500],
        "clf__max_depth": [2, 3, 4, 5, 6, 7],
        "clf__learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
        "clf__subsample": [0.6, 0.8, 1.0],
        "clf__min_samples_split": [2, 5, 10, 20],
        "clf__min_samples_leaf": [1, 2, 4, 8],
        "clf__max_features": ["sqrt", "log2", 0.3, 0.5],
        "clf__random_state": [42],
    }


def get_sklearn_param_space(model_type: str, target_pct: float) -> Dict[str, Any]:
    """Return sklearn param space keyed by pipeline step 'clf__' prefix."""
    if model_type == "rf":
        return _rf_space(target_pct)
    if model_type == "xgb":
        return _xgb_space(target_pct)
    if model_type == "lgb":
        return _lgb_space(target_pct)
    if model_type == "gb":
        return _gb_space(target_pct)
    raise ValueError(f"Unsupported model_type: {model_type}")


def suggest_optuna_params(
    trial, model_type: str, target_pct: float
) -> Dict[str, Any]:
    """Return params dict for a pipeline's 'clf' step, sampled via Optuna."""
    params: Dict[str, Any] = {}

    if model_type == "rf":
        params["n_estimators"] = trial.suggest_int("n_estimators", 100, 800, step=50)
        # Depth range target-aware
        if target_pct <= 0.01:
            params["max_depth"] = trial.suggest_categorical(
                "max_depth", [3, 5, 7, 10]
            )
            params["min_samples_leaf"] = trial.suggest_categorical(
                "min_samples_leaf", [2, 4, 8, 12]
            )
        elif target_pct <= 0.02:
            params["max_depth"] = trial.suggest_categorical(
                "max_depth", [5, 7, 10, 15]
            )
            params["min_samples_leaf"] = trial.suggest_categorical(
                "min_samples_leaf", [1, 2, 4, 8]
            )
        else:
            params["max_depth"] = trial.suggest_categorical(
                "max_depth", [7, 10, 15, 20, None]
            )
            params["min_samples_leaf"] = trial.suggest_categorical(
                "min_samples_leaf", [1, 2, 4, 8]
            )
        params["min_samples_split"] = trial.suggest_int(
            "min_samples_split", 2, 20, step=2
        )
        params["max_features"] = trial.suggest_categorical(
            "max_features", ["sqrt", "log2", 0.3, 0.5, 0.7]
        )
        params["bootstrap"] = trial.suggest_categorical("bootstrap", [True, False])
        params["class_weight"] = trial.suggest_categorical(
            "class_weight", ["balanced", "balanced_subsample", None]
        )
        params["random_state"] = 42

    elif model_type == "xgb":
        params["n_estimators"] = trial.suggest_int("n_estimators", 200, 800, step=50)
        params["max_depth"] = trial.suggest_int("max_depth", 3, 10)
        params["learning_rate"] = trial.suggest_float(
            "learning_rate", 0.005, 0.3, log=True
        )
        params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
        params["colsample_bytree"] = trial.suggest_float(
            "colsample_bytree", 0.5, 1.0
        )
        params["gamma"] = trial.suggest_float("gamma", 0.0, 2.0)
        params["reg_alpha"] = trial.suggest_float("reg_alpha", 0.0, 10.0)
        params["reg_lambda"] = trial.suggest_float("reg_lambda", 0.0, 10.0)
        params["scale_pos_weight"] = trial.suggest_float(
            "scale_pos_weight", 1.0, 5.0
        )
        params["random_state"] = 42
        # Stability defaults
        params["eval_metric"] = "logloss"
        params["n_jobs"] = -1
        params["use_label_encoder"] = False

    elif model_type == "lgb":
        params["n_estimators"] = trial.suggest_int("n_estimators", 300, 2000, step=100)
        params["max_depth"] = trial.suggest_categorical(
            "max_depth", [-1, 3, 5, 7, 10, 12]
        )
        params["learning_rate"] = trial.suggest_float(
            "learning_rate", 0.005, 0.3, log=True
        )
        params["num_leaves"] = trial.suggest_categorical(
            "num_leaves", [20, 31, 63, 127, 255]
        )
        params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
        params["colsample_bytree"] = trial.suggest_float(
            "colsample_bytree", 0.5, 1.0
        )
        params["reg_alpha"] = trial.suggest_float("reg_alpha", 0.0, 10.0)
        params["reg_lambda"] = trial.suggest_float("reg_lambda", 0.0, 10.0)
        params["min_child_samples"] = trial.suggest_int("min_child_samples", 5, 50)
        params["is_unbalance"] = trial.suggest_categorical(
            "is_unbalance", [True, False]
        )
        params["random_state"] = 42
        params["verbosity"] = -1
        params["n_jobs"] = -1

    elif model_type == "gb":
        params["n_estimators"] = trial.suggest_int("n_estimators", 100, 500, step=50)
        params["max_depth"] = trial.suggest_int("max_depth", 2, 7)
        params["learning_rate"] = trial.suggest_float(
            "learning_rate", 0.01, 0.3, log=True
        )
        params["subsample"] = trial.suggest_categorical("subsample", [0.6, 0.8, 1.0])
        params["min_samples_split"] = trial.suggest_int(
            "min_samples_split", 2, 20, step=2
        )
        params["min_samples_leaf"] = trial.suggest_categorical(
            "min_samples_leaf", [1, 2, 4, 8]
        )
        params["max_features"] = trial.suggest_categorical(
            "max_features", ["sqrt", "log2", 0.3, 0.5]
        )
        params["random_state"] = 42

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return params
