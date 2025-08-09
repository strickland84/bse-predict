"""Optuna-based hyperparameter tuner using Postgres storage and project datasets.

Key features:
- Uses MultiTargetFeatureEngineer to keep feature parity with production
- Prevents leakage via PurgedTimeSeriesSplit with configurable embargo_hours
- Evaluates with f1_weighted (primary); also computes accuracy and MCC on a final holdout split
- Persists best pipeline as a single pickle artifact under models_optimized/
- Records metadata into Postgres tables: model_tuning_history and model_registry
- Does NOT change production prediction behavior; artifacts are separate and inactive by default

CLI example:
    python -m src.tuning.optuna_tuner --symbol BTC/USDT --target 0.02 --model rf --trials 50
"""

from __future__ import annotations

import argparse
import json
import pickle
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd

# Suppress sklearn warnings about constant features
warnings.filterwarnings('ignore', message='Features .* are constant', category=UserWarning)
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import warnings

import lightgbm as lgb
import xgboost as xgb

from sqlalchemy import text

from src.data.feature_engineer import MultiTargetFeatureEngineer
from src.database.operations import DatabaseOperations
from src.tuning.purged_tscv import PurgedTimeSeriesSplit
from src.tuning.search_spaces import suggest_optuna_params
from src.utils.config import config
from src.utils.logger import get_logger
import os

logger = get_logger(__name__)


@dataclass
class TuningConfig:
    n_trials: int
    cv_folds: int
    embargo_hours: int
    scoring: str
    per_model_time_limit_min: int
    deploy_if_better: bool
    min_improvement_pct: float
    study_storage: Optional[str]


class SafeSelectKBest(BaseEstimator, TransformerMixin):
    def __init__(self, k: int = 10, score_func=f_classif):
        self.k = k
        self.score_func = score_func
        self.support_ = None
        self.selected_idx_ = None
        self.k_eff_ = None
        self.scores_ = None

    def fit(self, X, y):
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        y_arr = np.asarray(y)
        n_features = X_arr.shape[1]
        k_eff = max(1, min(int(self.k), n_features))

        # If the training target is single-class in this fold, fall back to variance-based selection
        uniq = np.unique(y_arr)
        if uniq.shape[0] < 2:
            scores = np.var(X_arr, axis=0)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                res = self.score_func(X_arr, y_arr)
            scores = res[0] if isinstance(res, tuple) else res
            scores = np.asarray(scores, dtype=float)
            scores[~np.isfinite(scores)] = -np.inf

        # Select top-k indices (stable order)
        idx = np.argsort(scores)[-k_eff:]
        idx = np.sort(idx)

        support = np.zeros(n_features, dtype=bool)
        support[idx] = True

        self.support_ = support
        self.selected_idx_ = idx
        self.k_eff_ = k_eff
        self.scores_ = scores
        return self

    def transform(self, X):
        if self.support_ is None:
            raise RuntimeError("SafeSelectKBest is not fitted")
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        return X_arr[:, self.selected_idx_]

    def get_support(self, indices: bool = False):
        if indices:
            return self.selected_idx_
        return self.support_


class OptunaHyperparameterTuner:
    def __init__(self):
        # Create database operations instance with correct URL
        database_url = os.environ.get("DATABASE_URL") or config.database_url
        self.db_ops = DatabaseOperations(database_url)
        
        self.assets = config.assets
        self.target_percentages = config.target_percentages
        self.models_dir = Path("models_optimized")
        self.models_dir.mkdir(exist_ok=True)
        self.tuning_cfg = TuningConfig(
            n_trials=int(config.get_nested("ml.tuning.n_trials", 50)),
            cv_folds=int(config.get_nested("ml.tuning.cv_folds", 5)),
            embargo_hours=int(config.get_nested("ml.tuning.embargo_hours", 72)),
            scoring=str(config.get_nested("ml.tuning.scoring", "f1_weighted")),
            per_model_time_limit_min=int(
                config.get_nested("ml.tuning.per_model_time_limit_min", 30)
            ),
            deploy_if_better=bool(config.get_nested("ml.tuning.deploy_if_better", False)),
            min_improvement_pct=float(config.get_nested("ml.tuning.min_improvement_pct", 3.0)),
            study_storage=config.get_nested("ml.tuning.study_storage"),
        )

    def _build_classifier(self, model_type: str, params: Dict[str, Any]):
        if model_type == "rf":
            return RandomForestClassifier(n_jobs=-1, **params)
        if model_type == "xgb":
            params = dict(params)
            params.setdefault("eval_metric", "logloss")
            params.setdefault("n_jobs", -1)
            params.setdefault("use_label_encoder", False)
            return xgb.XGBClassifier(**params)
        if model_type == "lgb":
            params = dict(params)
            params.setdefault("n_jobs", -1)
            params.setdefault("verbosity", -1)
            return lgb.LGBMClassifier(**params)
        if model_type == "gb":
            return GradientBoostingClassifier(**params)
        raise ValueError(f"Unsupported model_type: {model_type}")

    def _cv_score_manual(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        ts: pd.Series,
        model_type: str,
        params: Dict[str, Any],
        k_features: int,
    ) -> float:
        splitter = PurgedTimeSeriesSplit(
            n_splits=self.tuning_cfg.cv_folds, embargo_hours=self.tuning_cfg.embargo_hours
        )

        scores: List[float] = []
        k = max(1, min(k_features, X.shape[1]))
        for train_idx, test_idx in splitter.split(X.values, timestamps=ts):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            pipe = Pipeline(
                steps=[
                    ("select", SafeSelectKBest(k=min(k, X_train.shape[1]), score_func=f_classif)),
                    ("clf", self._build_classifier(model_type, params)),
                ]
            )
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            score = f1_score(y_test, y_pred, average="weighted")
            if np.isnan(score):
                score = 0.0
            scores.append(score)

        if not scores:
            return 0.0
        return float(np.mean(scores))

    def _objective(
        self,
        trial: optuna.trial.Trial,
        X: pd.DataFrame,
        y: pd.Series,
        ts: pd.Series,
        model_type: str,
        target_pct: float,
    ) -> float:
        max_k = min(120, X.shape[1])
        min_k = min(20, max_k)
        step = 1
        k = trial.suggest_int("kbest_k", min_k, max_k, step=step)

        params = suggest_optuna_params(trial, model_type, target_pct)
        score = self._cv_score_manual(X, y, ts, model_type, params, k)

        trial.report(score, step=1)
        if trial.should_prune():
            raise optuna.TrialPruned()

        return score

    def _persist_tuning_history(
        self,
        symbol: str,
        target_pct: float,
        model_type: str,
        study_name: str,
        engine: str,
        n_trials: int,
        best_score: Optional[float],
        best_params: Dict[str, Any],
        status: str = "completed",
    ) -> None:
        try:
            with self.db_ops.db.get_session() as session:
                session.execute(
                    text(
                        """
                    INSERT INTO model_tuning_history
                        (symbol, target_pct, model_type, study_name, engine, n_trials, best_score, best_params, started_at, completed_at, status)
                    VALUES
                        (:symbol, :target_pct, :model_type, :study_name, :engine, :n_trials, :best_score, :best_params, NOW(), NOW(), :status)
                    """
                    ),
                    {
                        "symbol": symbol,
                        "target_pct": float(target_pct),
                        "model_type": model_type,
                        "study_name": study_name,
                        "engine": engine,
                        "n_trials": int(n_trials),
                        "best_score": float(best_score) if best_score is not None else None,
                        "best_params": json.dumps(best_params),
                        "status": status,
                    },
                )
                session.commit()
        except Exception as e:
            logger.error(f"Failed to persist model_tuning_history: {e}")

    def _persist_model_registry(
        self,
        symbol: str,
        target_pct: float,
        model_type: str,
        source: str,
        file_path: str,
        params: Dict[str, Any],
        cv_score: Optional[float],
        test_metrics: Dict[str, Any],
        is_active: bool = False,
        notes: Optional[str] = None,
    ) -> None:
        try:
            with self.db_ops.db.get_session() as session:
                session.execute(
                    text(
                        """
                    INSERT INTO model_registry
                        (symbol, target_pct, model_type, source, file_path, params, cv_score, test_metrics, is_active, trained_at, notes)
                    VALUES
                        (:symbol, :target_pct, :model_type, :source, :file_path, :params, :cv_score, :test_metrics, :is_active, NOW(), :notes)
                    """
                    ),
                    {
                        "symbol": symbol,
                        "target_pct": float(target_pct),
                        "model_type": model_type,
                        "source": source,
                        "file_path": file_path,
                        "params": json.dumps(params),
                        "cv_score": float(cv_score) if cv_score is not None else None,
                        "test_metrics": json.dumps(test_metrics),
                        "is_active": bool(is_active),
                        "notes": notes,
                    },
                )
                session.commit()
        except Exception as e:
            logger.error(f"Failed to persist model_registry: {e}")

    def _final_fit_and_evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_type: str,
        best_params: Dict[str, Any],
        k: int,
    ) -> Tuple[Pipeline, Dict[str, float]]:
        split_idx = int(len(X) * max(0.2, float(config.get_nested("ml.test_size", 0.2))))
        split_idx = max(split_idx, 1)
        train_end = len(X) - split_idx
        X_train, X_test = X.iloc[:train_end], X.iloc[train_end:]
        y_train, y_test = y.iloc[:train_end], y.iloc[train_end:]

        k_eff = max(1, min(k, X_train.shape[1]))
        pipe = Pipeline(
            steps=[
                ("select", SafeSelectKBest(k=k_eff, score_func=f_classif)),
                ("clf", self._build_classifier(model_type, best_params)),
            ]
        )
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="weighted", zero_division=0
        )
        mcc = matthews_corrcoef(y_test, y_pred)

        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_weighted": float(f1),
            "mcc": float(mcc),
            "test_samples": int(len(y_test)),
        }
        return pipe, metrics

    def _get_active_cv_score(self, symbol: str, target_pct: float) -> Optional[float]:
        """Fetch cv_score for the currently active model in model_registry."""
        try:
            with self.db_ops.db.get_session() as session:
                row = session.execute(
                    text(
                        """
                        SELECT cv_score 
                        FROM model_registry
                        WHERE symbol = :symbol 
                          AND target_pct = :target_pct
                          AND is_active = TRUE
                        ORDER BY trained_at DESC
                        LIMIT 1
                        """
                    ),
                    {"symbol": symbol, "target_pct": float(target_pct)},
                ).fetchone()
            if not row:
                return None
            # Row can be tuple-like or attr-like depending on dialect
            value = row[0] if not hasattr(row, "cv_score") else row.cv_score
            return float(value) if value is not None else None
        except Exception as e:
            logger.error(f"Failed to fetch active cv_score: {e}")
            return None

    def _activate_model(self, symbol: str, target_pct: float, file_path: str) -> None:
        """Activate a model in model_registry by file_path and deactivate others."""
        try:
            with self.db_ops.db.get_session() as session:
                session.execute(
                    text(
                        """
                        UPDATE model_registry
                        SET is_active = FALSE
                        WHERE symbol = :symbol AND target_pct = :target_pct
                        """
                    ),
                    {"symbol": symbol, "target_pct": float(target_pct)},
                )
                session.execute(
                    text(
                        """
                        UPDATE model_registry
                        SET is_active = TRUE
                        WHERE symbol = :symbol 
                          AND target_pct = :target_pct
                          AND file_path = :file_path
                        """
                    ),
                    {
                        "symbol": symbol,
                        "target_pct": float(target_pct),
                        "file_path": file_path,
                    },
                )
                session.commit()
        except Exception as e:
            logger.error(f"Failed to activate model in registry: {e}")

    def run(
        self, symbol: str, target_pct: float, model_type: str, n_trials_override: Optional[int] = None
    ) -> Dict[str, Any]:
        logger.info(
            f"🔧 Starting Optuna tuning for {symbol} {target_pct:.1%} [{model_type}]"
        )

        # Fetch data
        training_limit = config.get_nested('ml.training_data_limit', 4000)
        df = self.db_ops.get_latest_candles(symbol, "1h", limit=training_limit, include_futures=True)
        if df.empty or len(df) < 500:
            raise RuntimeError(f"Insufficient data for tuning: {len(df)} candles")

        # Build datasets (with timestamps)
        fe = MultiTargetFeatureEngineer([target_pct])
        datasets = fe.prepare_training_datasets_with_timestamps(df)
        if target_pct not in datasets:
            raise RuntimeError("Failed to prepare datasets for target_pct")

        X, y, ts = datasets[target_pct]
        if X is None or y is None or ts is None or len(X) < 200:
            raise RuntimeError("Insufficient samples after dataset preparation")

        # Study setup: prefer OPTUNA_STORAGE env, then config study_storage, else default to local SQLite.
        storage_url = os.environ.get("OPTUNA_STORAGE") or self.tuning_cfg.study_storage
        if not storage_url:
            storage_url = "sqlite:////app/data/optuna_studies.db"
        if storage_url.startswith("postgresql://"):
            storage_url = storage_url.replace("postgresql://", "postgresql+psycopg2://")
        logger.info(f"Using Optuna storage: {storage_url}")
        clean_symbol = symbol.replace("/", "_").replace("-", "_")
        study_name = f"hpo_{clean_symbol}_{target_pct:.3f}_{model_type}"

        sampler = optuna.samplers.TPESampler(seed=42)
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)

        try:
            study = optuna.create_study(
                direction="maximize",
                storage=storage_url,
                study_name=study_name,
                load_if_exists=True,
                sampler=sampler,
                pruner=pruner,
            )
        except RuntimeError as e:
            # Handle Optuna storage schema/version incompatibility by falling back to local SQLite
            msg = str(e).lower()
            if "schema" in msg or "table schema" in msg:
                logger.error(
                    f"Optuna storage schema incompatibility detected for '{storage_url}'. "
                    "Falling back to local SQLite at /app/data/optuna_studies.db"
                )
                fallback_path = Path("/app/data/optuna_studies.db")
                fallback_path.parent.mkdir(parents=True, exist_ok=True)
                storage_url = f"sqlite:///{fallback_path}"
                logger.info(f"Using Optuna fallback storage: {storage_url}")
                study = optuna.create_study(
                    direction="maximize",
                    storage=storage_url,
                    study_name=study_name,
                    load_if_exists=True,
                    sampler=sampler,
                    pruner=pruner,
                )
            else:
                raise

        n_trials = int(n_trials_override or self.tuning_cfg.n_trials)

        def _obj(trial: optuna.trial.Trial) -> float:
            return self._objective(trial, X, y, ts, model_type, target_pct)

        logger.info(
            f"🔍 Optimizing with {n_trials} trials, CV={self.tuning_cfg.cv_folds}, embargo={self.tuning_cfg.embargo_hours}h"
        )
        study.optimize(_obj, n_trials=n_trials, gc_after_trial=True)

        best_value = study.best_value if study.best_trial else None
        best_params = dict(study.best_trial.params) if study.best_trial else {}
        kbest_k = int(best_params.pop("kbest_k", min(60, X.shape[1])))

        # Final fit on holdout split
        pipeline, test_metrics = self._final_fit_and_evaluate(
            X, y, model_type, best_params, kbest_k
        )

        # Persist artifacts
        filename = f"{model_type}_{clean_symbol}_{target_pct:.3f}_optuna.pkl"
        filepath = self.models_dir / filename
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "pipeline": pipeline,
                    "symbol": symbol,
                    "target_pct": float(target_pct),
                    "model_type": model_type,
                    "trained_at": datetime.now(),
                    "cv_score": float(best_value) if best_value is not None else None,
                    "test_metrics": test_metrics,
                    "params": best_params,
                    "kbest_k": int(kbest_k),
                    "feature_version": fe.feature_version,
                    "feature_columns": list(X.columns),
                    "source": "optuna",
                },
                f,
            )
        logger.info(f"💾 Saved tuned model to {filepath}")

        # Write JSON sidecar summary
        summary = {
            "symbol": symbol,
            "target_pct": float(target_pct),
            "model_type": model_type,
            "cv_score": float(best_value) if best_value is not None else None,
            "test_metrics": test_metrics,
            "params": best_params,
            "kbest_k": kbest_k,
            "trained_at": datetime.now().isoformat(),
            "feature_version": fe.feature_version,
            "study_name": study_name,
            "file_path": str(filepath),
        }
        with open(self.models_dir / f"{filename}.json", "w") as jf:
            json.dump(summary, jf, indent=2)

        # Persist DB records
        self._persist_tuning_history(
            symbol=symbol,
            target_pct=target_pct,
            model_type=model_type,
            study_name=study_name,
            engine="optuna",
            n_trials=n_trials,
            best_score=best_value,
            best_params=best_params,
            status="completed",
        )

        # Registry insert (inactive by default)
        self._persist_model_registry(
            symbol=symbol,
            target_pct=target_pct,
            model_type=model_type,
            source="optuna",
            file_path=str(filepath),
            params=best_params,
            cv_score=best_value,
            test_metrics=test_metrics,
            is_active=False,
            notes="Created by Optuna tuner; inactive by default",
        )
        
        # Also save to model_training_history for consistency with regular training
        try:
            # Get feature importance from the pipeline's classifier
            if hasattr(pipeline.named_steps['clf'], 'feature_importances_'):
                # Get selected feature names after SafeSelectKBest selection
                select_step = pipeline.named_steps['select']
                selected_mask = select_step.get_support()
                selected_features = [f for f, mask in zip(X.columns, selected_mask) if mask]
                
                feature_importance = pd.DataFrame({
                    'feature': selected_features,
                    'importance': pipeline.named_steps['clf'].feature_importances_
                }).sort_values('importance', ascending=False)
                top_features = feature_importance.head(10).to_dict('records')
            else:
                top_features = []
            
            clean_symbol = symbol.replace('/', '_').replace('-', '_')
            filename = f"{model_type}_{clean_symbol}_{target_pct:.3f}_optuna_model.pkl"
            
            # Build model config for storage
            model_config = {
                'model_type': model_type,
                'hyperparameters': best_params,
                'kbest_k': kbest_k,
                'tuning_engine': 'optuna',
                'n_trials': n_trials,
                'study_name': study_name
            }
            
            query = text("""
                INSERT INTO model_training_history 
                (symbol, target_pct, trained_at, model_filename, training_samples, features_count,
                 date_range_start, date_range_end, price_range_min, price_range_max,
                 target_distribution, cv_accuracy, cv_std, final_accuracy, precision, recall, f1_score,
                 top_features, model_config)
                VALUES (:symbol, :target_pct, :trained_at, :model_filename, :training_samples, :features_count,
                        :date_range_start, :date_range_end, :price_range_min, :price_range_max,
                        :target_distribution, :cv_accuracy, :cv_std, :final_accuracy, :precision, :recall, :f1_score,
                        :top_features, :model_config)
            """)
            
            # Calculate target distribution from y
            target_dist = y.value_counts(normalize=True).to_dict()
            target_dist = {int(k): float(v) for k, v in target_dist.items()}
            
            with self.db_ops.db.get_session() as session:
                session.execute(query, {
                    'symbol': symbol,
                    'target_pct': float(target_pct),
                    'trained_at': datetime.now(),
                    'model_filename': filename,
                    'training_samples': int(len(X)),
                    'features_count': int(kbest_k),  # Number of features after selection
                    'date_range_start': df['timestamp'].min(),
                    'date_range_end': df['timestamp'].max(),
                    'price_range_min': float(df['close'].min()),
                    'price_range_max': float(df['close'].max()),
                    'target_distribution': json.dumps(target_dist),
                    'cv_accuracy': float(best_value) if best_value else 0.5,
                    'cv_std': 0.0,  # Not calculated in Optuna tuning currently
                    'final_accuracy': float(test_metrics.get('accuracy', 0.5)),
                    'precision': float(test_metrics.get('precision', 0.5)),
                    'recall': float(test_metrics.get('recall', 0.5)),
                    'f1_score': float(test_metrics.get('f1', 0.5)),
                    'top_features': json.dumps(top_features),
                    'model_config': json.dumps(model_config)
                })
                session.commit()
                
            logger.info(f"Training history saved for {symbol} {target_pct:.1%} (Optuna-tuned {model_type})")
            
        except Exception as e:
            logger.error(f"Error saving training history for tuned model: {e}")
            raise RuntimeError(f"Failed to save model training history: {e}") from e

        # Copy optimized model to regular models directory for immediate use
        try:
            regular_model_path = Path("models") / f"rf_{symbol.replace('/', '_')}_{target_pct:.3f}_model.pkl"
            regular_model_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save just the model (not the full pipeline) to match expected format
            model_only = {
                'model': pipeline.named_steps['clf'],  # Use the pipeline variable from above
                'features': list(X.columns),  # Use the feature columns from training
                'accuracy': best_value if best_value else 0.5,
                'trained_at': datetime.now().isoformat(),
                'data_version': 'tuned_optuna',
                'model_type': 'rf',
                'hyperparameters': best_params
            }
            
            with open(regular_model_path, 'wb') as f:
                pickle.dump(model_only, f)
            logger.info(f"✅ Saved tuned model to regular models directory: {regular_model_path}")
            print(f"✅ Saved tuned model to regular models directory: {regular_model_path}")
        except Exception as e:
            logger.error(f"Failed to save tuned model to regular directory: {e}")
            print(f"❌ Failed to save tuned model to regular directory: {e}")
        
        # Optional activation gate (safe, non-breaking by default)
        if self.tuning_cfg.deploy_if_better:
            prev = self._get_active_cv_score(symbol, target_pct)
            if prev is None:
                self._activate_model(symbol, target_pct, str(filepath))
                logger.info("Activated first model in registry for this symbol/target")
            else:
                if best_value is not None and prev > 0:
                    improvement = ((best_value - prev) / abs(prev)) * 100.0
                    if improvement >= self.tuning_cfg.min_improvement_pct:
                        self._activate_model(symbol, target_pct, str(filepath))
                        logger.info(
                            f"Activated tuned model (improved {improvement:.2f}% over {prev:.4f})"
                        )
                    else:
                        logger.info(
                            f"Not activating tuned model (improvement {improvement:.2f}% < {self.tuning_cfg.min_improvement_pct:.2f}%)"
                        )

        result = {
            "file_path": str(filepath),
            "cv_score": float(best_value) if best_value is not None else None,
            "test_metrics": test_metrics,
            "best_params": best_params,
            "kbest_k": int(kbest_k),
            "study_name": study_name,
        }
        logger.info(f"✅ Tuning finished: {json.dumps(result, indent=2)}")
        return result


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Optuna HPO for a given symbol/target/model")
    p.add_argument("--symbol", required=True, help="Trading symbol, e.g., BTC/USDT")
    p.add_argument("--target", required=True, type=float, help="Target pct, e.g., 0.02")
    p.add_argument("--model", required=True, choices=["rf", "xgb", "lgb", "gb"], help="Model type")
    p.add_argument("--trials", type=int, default=None, help="Override number of trials")
    return p.parse_args()


def main():
    args = _parse_args()
    tuner = OptunaHyperparameterTuner()
    tuner.run(symbol=args.symbol, target_pct=args.target, model_type=args.model, n_trials_override=args.trials)


if __name__ == "__main__":
    main()
