"""Initial schema creation (idempotent).

- Core tables: ohlcv_data, feature_cache, predictions, prediction_outcomes,
  model_performance, telegram_reports, system_health
- Additional tables used by app: system_errors, model_training_history
- Indexes and materialized view
- TimescaleDB hypertables for ohlcv_data and feature_cache (best-effort)

This migration is safe to run on databases that were initialized via sql/init.sql
since all statements are guarded with IF NOT EXISTS or wrapped DO blocks.
"""

from alembic import op
import sqlalchemy as sa

# Revision identifiers, used by Alembic.
revision = "1a2b3c4d5e6f"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Try to enable timescaledb extension, but do not fail if lacking privileges
    op.execute(
        """
DO $$
BEGIN
    BEGIN
        CREATE EXTENSION IF NOT EXISTS timescaledb;
    EXCEPTION
        WHEN insufficient_privilege THEN
            RAISE NOTICE 'Insufficient privilege to create extension timescaledb, skipping';
        WHEN others THEN
            RAISE NOTICE 'Could not create extension timescaledb: %', SQLERRM;
    END;
END;
$$;
"""
    )

    # ohlcv_data
    op.execute(
        """
CREATE TABLE IF NOT EXISTS ohlcv_data (
    id BIGSERIAL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    open DECIMAL(20,8) NOT NULL,
    high DECIMAL(20,8) NOT NULL,
    low DECIMAL(20,8) NOT NULL,
    close DECIMAL(20,8) NOT NULL,
    volume DECIMAL(20,8) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (symbol, timeframe, timestamp)
);
"""
    )

    # feature_cache
    op.execute(
        """
CREATE TABLE IF NOT EXISTS feature_cache (
    id BIGSERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    feature_set JSONB NOT NULL,
    feature_version TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(timestamp, symbol, feature_version)
);
"""
    )

    # predictions
    op.execute(
        """
CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    target_pct DECIMAL(4,3) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    model_name TEXT NOT NULL,
    prediction_class INTEGER NOT NULL,
    probability DECIMAL(5,4) NOT NULL,
    confidence DECIMAL(5,4) NOT NULL,
    features_used JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(symbol, target_pct, timestamp, model_name)
);
"""
    )

    # prediction_outcomes (unique constraint added in later migration)
    op.execute(
        """
CREATE TABLE IF NOT EXISTS prediction_outcomes (
    id BIGSERIAL PRIMARY KEY,
    prediction_id BIGINT REFERENCES predictions(id),
    actual_outcome INTEGER,
    target_hit_timestamp TIMESTAMPTZ,
    time_to_target_hours DECIMAL(6,2),
    max_favorable_move DECIMAL(6,4),
    max_adverse_move DECIMAL(6,4),
    completed_at TIMESTAMPTZ DEFAULT NOW()
);
"""
    )

    # model_performance
    op.execute(
        """
CREATE TABLE IF NOT EXISTS model_performance (
    id BIGSERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    target_pct DECIMAL(4,3) NOT NULL,
    model_name TEXT NOT NULL,
    evaluation_date DATE NOT NULL,
    accuracy DECIMAL(5,4),
    precision_up DECIMAL(5,4),
    precision_down DECIMAL(5,4),
    recall_up DECIMAL(5,4),
    recall_down DECIMAL(5,4),
    f1_up DECIMAL(5,4),
    f1_down DECIMAL(5,4),
    total_predictions INTEGER,
    avg_time_to_target_hours DECIMAL(6,2),
    hit_rate DECIMAL(5,4),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(symbol, target_pct, model_name, evaluation_date)
);
"""
    )

    # telegram_reports
    op.execute(
        """
CREATE TABLE IF NOT EXISTS telegram_reports (
    id BIGSERIAL PRIMARY KEY,
    report_type TEXT NOT NULL,
    content TEXT NOT NULL,
    sent_at TIMESTAMPTZ DEFAULT NOW(),
    success BOOLEAN DEFAULT FALSE,
    error_message TEXT
);
"""
    )

    # system_health
    op.execute(
        """
CREATE TABLE IF NOT EXISTS system_health (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    cpu_percent DECIMAL(5,2),
    memory_percent DECIMAL(5,2),
    disk_percent DECIMAL(5,2),
    prediction_latency_ms INTEGER,
    data_freshness_minutes INTEGER,
    active_models INTEGER
);
"""
    )

    # system_errors (from init_db.py)
    op.execute(
        """
CREATE TABLE IF NOT EXISTS system_errors (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    logger_name VARCHAR(200) NOT NULL,
    level VARCHAR(20) NOT NULL,
    message TEXT NOT NULL,
    exception TEXT,
    traceback TEXT,
    module VARCHAR(200),
    function VARCHAR(200),
    line_number INTEGER,
    thread_name VARCHAR(100),
    process_id INTEGER,
    hostname VARCHAR(200)
);
"""
    )

    # model_training_history (from init_db.py)
    op.execute(
        """
CREATE TABLE IF NOT EXISTS model_training_history (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    target_pct DECIMAL(4,3) NOT NULL,
    trained_at TIMESTAMPTZ NOT NULL,
    model_filename VARCHAR(100) NOT NULL,
    training_samples INTEGER NOT NULL,
    features_count INTEGER NOT NULL,
    date_range_start TIMESTAMPTZ NOT NULL,
    date_range_end TIMESTAMPTZ NOT NULL,
    price_range_min DECIMAL(20,8) NOT NULL,
    price_range_max DECIMAL(20,8) NOT NULL,
    target_distribution JSONB NOT NULL,
    cv_accuracy DECIMAL(5,4) NOT NULL,
    cv_std DECIMAL(5,4) NOT NULL,
    final_accuracy DECIMAL(5,4) NOT NULL,
    precision DECIMAL(5,4) NOT NULL,
    recall DECIMAL(5,4) NOT NULL,
    f1_score DECIMAL(5,4) NOT NULL,
    top_features JSONB NOT NULL,
    model_config JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
"""
    )

    # Indexes
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_time ON ohlcv_data(symbol, timestamp DESC);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_feature_cache_symbol_time ON feature_cache(symbol, timestamp DESC);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_predictions_symbol_target_time ON predictions(symbol, target_pct, timestamp DESC);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_predictions_features ON predictions USING GIN (features_used);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_outcomes_prediction ON prediction_outcomes(prediction_id);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_performance_symbol_target_model ON model_performance(symbol, target_pct, model_name, evaluation_date DESC);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_health_timestamp ON system_health(timestamp DESC);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_errors_timestamp ON system_errors(timestamp DESC);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_errors_level ON system_errors(level);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_errors_logger ON system_errors(logger_name);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_training_history_symbol_target ON model_training_history(symbol, target_pct, trained_at DESC);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_training_history_features ON model_training_history USING GIN (top_features);"
    )

    # Materialized view and refresh function
    op.execute(
        """
CREATE MATERIALIZED VIEW IF NOT EXISTS recent_ohlcv AS
SELECT * FROM ohlcv_data 
WHERE timestamp >= NOW() - INTERVAL '7 days'
ORDER BY symbol, timestamp DESC;
"""
    )
    op.execute(
        """
CREATE UNIQUE INDEX IF NOT EXISTS idx_recent_ohlcv_unique ON recent_ohlcv(symbol, timeframe, timestamp);
"""
    )
    op.execute(
        """
CREATE OR REPLACE FUNCTION refresh_recent_ohlcv()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY recent_ohlcv;
END;
$$ LANGUAGE plpgsql;
"""
    )

    # Hypertables - best effort wrappers to avoid breaking if ext is not available
    op.execute(
        """
DO $$
BEGIN
    BEGIN
        PERFORM create_hypertable('ohlcv_data', 'timestamp', 'symbol', 3, if_not_exists => TRUE);
    EXCEPTION
        WHEN undefined_function THEN
            RAISE NOTICE 'TimescaleDB not available, skipping hypertable for ohlcv_data';
        WHEN others THEN
            RAISE NOTICE 'Could not create hypertable for ohlcv_data: %', SQLERRM;
    END;

    BEGIN
        PERFORM create_hypertable('feature_cache', 'timestamp', 'symbol', 3, if_not_exists => TRUE);
    EXCEPTION
        WHEN undefined_function THEN
            RAISE NOTICE 'TimescaleDB not available, skipping hypertable for feature_cache';
        WHEN others THEN
            RAISE NOTICE 'Could not create hypertable for feature_cache: %', SQLERRM;
    END;
END;
$$;
"""
    )


def downgrade() -> None:
    # Non-destructive downgrade: keep schema to avoid breaking running systems
    # If you truly need to drop objects, implement explicit drops here.
    pass
