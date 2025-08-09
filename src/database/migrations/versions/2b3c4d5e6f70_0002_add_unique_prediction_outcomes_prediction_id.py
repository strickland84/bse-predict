"""Add UNIQUE constraint on prediction_outcomes(prediction_id) safely.

- Checks for duplicates first; if duplicates exist, logs a NOTICE and skips adding the constraint.
- If no duplicates and constraint missing, adds the constraint.
- Idempotent and safe to run multiple times.
"""

from alembic import op
import sqlalchemy as sa

# Revision identifiers, used by Alembic.
revision = "2b3c4d5e6f70"
down_revision = "1a2b3c4d5e6f"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
DO $$
DECLARE
    has_duplicates boolean;
    constraint_exists boolean;
BEGIN
    -- Check for duplicate prediction_id rows which would block the UNIQUE constraint
    SELECT EXISTS (
        SELECT prediction_id
        FROM prediction_outcomes
        WHERE prediction_id IS NOT NULL
        GROUP BY prediction_id
        HAVING COUNT(*) > 1
        LIMIT 1
    ) INTO has_duplicates;

    IF has_duplicates THEN
        RAISE NOTICE 'Duplicate prediction_outcomes.prediction_id rows detected. Skipping UNIQUE constraint. Run scripts/fix_prediction_outcomes_constraint.py and rerun migrations.';
        RETURN;
    END IF;

    -- Check if the unique constraint already exists by name
    SELECT EXISTS (
        SELECT 1
        FROM information_schema.table_constraints tc
        WHERE tc.table_name = 'prediction_outcomes'
          AND tc.constraint_type = 'UNIQUE'
          AND tc.constraint_name = 'prediction_outcomes_prediction_id_key'
    ) INTO constraint_exists;

    IF NOT constraint_exists THEN
        BEGIN
            ALTER TABLE prediction_outcomes
            ADD CONSTRAINT prediction_outcomes_prediction_id_key UNIQUE (prediction_id);
        EXCEPTION
            WHEN duplicate_table THEN
                -- Already exists, ignore
                NULL;
            WHEN others THEN
                RAISE NOTICE 'Could not add UNIQUE constraint on prediction_outcomes(prediction_id): %', SQLERRM;
        END;
    END IF;
END;
$$;
"""
    )


def downgrade() -> None:
    # Non-destructive: do not remove constraints automatically to avoid breaking systems
    # If you must drop the constraint, uncomment below.
    # op.execute("ALTER TABLE IF EXISTS prediction_outcomes DROP CONSTRAINT IF EXISTS prediction_outcomes_prediction_id_key;")
    pass
