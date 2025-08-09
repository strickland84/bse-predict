"""Formalize futures_data table and TimescaleDB hypertable (idempotent).

- Creates futures_data table if it doesn't exist (schema matching current runtime usage)
- Adds UNIQUE(symbol, timestamp) and index on (symbol, timestamp DESC)
- Attempts to enable TimescaleDB and convert to hypertable (best-effort, non-fatal)
"""

from alembic import op
import sqlalchemy as sa

# Revision identifiers, used by Alembic.
revision = "3c4d5e6f7081"
down_revision = "2b3c4d5e6f70"
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

    # Create futures_data table if not exists (mirrors current runtime table definition)
    op.execute(
        """
CREATE TABLE IF NOT EXISTS futures_data (
    id SERIAL,
    symbol TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    open_interest DECIMAL(20, 8),
    open_interest_value DECIMAL(20, 8),
    funding_rate DECIMAL(10, 8),
    mark_price DECIMAL(20, 8),
    top_trader_ratio DECIMAL(10, 4),
    taker_buy_sell_ratio DECIMAL(10, 4),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (id, timestamp),
    UNIQUE(symbol, timestamp)
);
"""
    )

    # Index for query performance
    op.execute(
        """
CREATE INDEX IF NOT EXISTS idx_futures_symbol_time 
ON futures_data (symbol, timestamp DESC);
"""
    )

    # Hypertable conversion - best effort with data migration if needed
    op.execute(
        """
DO $$
BEGIN
    -- Wrap in inner block to catch errors if TimescaleDB is not available
    BEGIN
        -- Only attempt if not already a hypertable
        IF NOT EXISTS (
            SELECT 1 FROM _timescaledb_catalog.hypertable 
            WHERE table_name = 'futures_data'
        ) THEN
            -- If table has data, migrate it; otherwise create hypertable normally
            IF EXISTS (SELECT 1 FROM futures_data LIMIT 1) THEN
                PERFORM create_hypertable('futures_data', 'timestamp', migrate_data => TRUE, if_not_exists => TRUE);
            ELSE
                PERFORM create_hypertable('futures_data', 'timestamp', if_not_exists => TRUE);
            END IF;
        END IF;
    EXCEPTION
        WHEN undefined_table THEN
            -- _timescaledb_catalog not available: extension missing
            RAISE NOTICE 'TimescaleDB catalog not available, skipping hypertable for futures_data';
        WHEN invalid_schema_name THEN
            RAISE NOTICE 'TimescaleDB schema not available, skipping hypertable for futures_data';
        WHEN undefined_function THEN
            RAISE NOTICE 'TimescaleDB create_hypertable function not available, skipping';
        WHEN others THEN
            RAISE NOTICE 'Could not create hypertable for futures_data: %', SQLERRM;
    END;
END;
$$;
"""
    )


def downgrade() -> None:
    # Non-destructive downgrade to avoid breaking running systems
    # If you truly need to drop objects, implement explicit drops here.
    pass
