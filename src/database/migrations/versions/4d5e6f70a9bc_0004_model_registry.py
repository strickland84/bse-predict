"""Add model_registry and model_tuning_history tables (idempotent).

- model_registry: tracks trained models (baseline/optimized), params, metrics, and activation state
- model_tuning_history: optional summary records for HPO studies/runs

This migration is safe to run repeatedly (IF NOT EXISTS guards).
"""

from alembic import op
import sqlalchemy as sa

# Revision identifiers, used by Alembic.
revision = "4d5e6f70a9bc"
down_revision = "3c4d5e6f7081"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # model_registry
    op.execute(
        """
CREATE TABLE IF NOT EXISTS model_registry (
    id BIGSERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    target_pct DECIMAL(4,3) NOT NULL,
    model_type TEXT NOT NULL,                 -- e.g., 'rf', 'xgb', 'lgb', 'gb'
    source TEXT NOT NULL,                     -- e.g., 'baseline', 'optuna', 'random', 'grid'
    file_path TEXT NOT NULL,                  -- where the serialized model is stored
    params JSONB NOT NULL,                    -- model hyperparameters
    cv_score DECIMAL(6,4),                    -- best CV score (primary metric)
    test_metrics JSONB,                       -- final/holdout metrics (accuracy, f1, mcc, etc.)
    is_active BOOLEAN DEFAULT FALSE,          -- whether this model is the selected one (for optional usage)
    trained_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    notes TEXT
);
"""
    )

    # Indexes for registry
    op.execute(
        """
CREATE INDEX IF NOT EXISTS idx_model_registry_symbol_target_trained 
ON model_registry (symbol, target_pct, trained_at DESC);
"""
    )
    op.execute(
        """
CREATE INDEX IF NOT EXISTS idx_model_registry_symbol_target_active 
ON model_registry (symbol, target_pct) WHERE is_active = TRUE;
"""
    )
    op.execute(
        """
CREATE INDEX IF NOT EXISTS idx_model_registry_model_type 
ON model_registry (model_type);
"""
    )
    op.execute(
        """
CREATE INDEX IF NOT EXISTS idx_model_registry_params_gin 
ON model_registry USING GIN (params);
"""
    )

    # model_tuning_history (optional high-level study summaries)
    op.execute(
        """
CREATE TABLE IF NOT EXISTS model_tuning_history (
    id BIGSERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    target_pct DECIMAL(4,3) NOT NULL,
    model_type TEXT NOT NULL,
    study_name TEXT NOT NULL,
    engine TEXT NOT NULL,             -- 'optuna' | 'random' | 'grid'
    n_trials INTEGER NOT NULL,
    best_score DECIMAL(6,4),
    best_params JSONB,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    status TEXT                       -- 'completed' | 'aborted' | 'failed' | 'running'
);
"""
    )

    op.execute(
        """
CREATE INDEX IF NOT EXISTS idx_tuning_history_symbol_target_time 
ON model_tuning_history (symbol, target_pct, started_at DESC);
"""
    )
    op.execute(
        """
CREATE INDEX IF NOT EXISTS idx_tuning_history_best_params_gin 
ON model_tuning_history USING GIN (best_params);
"""
    )


def downgrade() -> None:
    # Non-destructive downgrade to avoid breaking running systems
    # If you truly need to drop objects, implement explicit drops here.
    pass
