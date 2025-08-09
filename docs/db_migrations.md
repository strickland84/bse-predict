# Database Migrations with Alembic

This project uses Alembic for safe, versioned, idempotent database schema management. Migrations are designed to work without breaking existing setups and to play nicely with the current Docker dev environment which still mounts `sql/init.sql` for first-time DB initialization.

Key locations:
- Alembic config: `alembic.ini`
- Migration env: `src/database/migrations/env.py`
- Migration versions: `src/database/migrations/versions/`
- Programmatic runner: `src/database/migrations_runner.py`
- App initialization: `src/database/init_db.py` (now calls Alembic upgrade to head)

## Why Alembic?

- Versioned and auditable schema changes
- Idempotent operations using IF NOT EXISTS or guarded blocks
- Safe rollout that won’t break existing databases
- Works with TimescaleDB best-effort (extension and hypertables)
- Separation of DDL from runtime code

## Runtime behavior

On application startup, `src/main.py` calls `init_database()`, which now:
1. Tests DB connectivity.
2. Runs Alembic `upgrade head` programmatically.
3. Verifies required tables exist and logs the status.

This ensures the schema is always up-to-date with the latest migrations.

## Developer workflow

Makefile helpers:
- Create a new migration:
  - `make db-revision NAME="add new table or column"`
  - This generates a new Python migration in `src/database/migrations/versions/`.
  - Write your DDL inside `upgrade()` as `op.execute("""...""")` statements, using `IF NOT EXISTS` or guarded `DO $$ ... $$;` blocks when appropriate.

- Apply migrations:
  - `make db-upgrade` (to head)
  - `make db-upgrade REV=1a2b3c4d5e6f` (to a specific revision)

These targets run inside the running app container if available; otherwise they run locally with your environment.

Manual programmatic run:
- From Python:  
  ```python
  from src.database.migrations_runner import run_migrations
  run_migrations("postgresql://user:pass@host:5432/dbname")  # or omit argument if DATABASE_URL env is set
  ```

## Current migration set

- 0001_initial_schema
  - Creates core tables: `ohlcv_data`, `feature_cache`, `predictions`, `prediction_outcomes`, `model_performance`, `telegram_reports`, `system_health`.
  - Adds additional tables used by the app: `system_errors`, `model_training_history`.
  - Creates all required indexes and materialized view `recent_ohlcv` (+ refresh function).
  - Best-effort TimescaleDB extension setup and hypertable conversion for `ohlcv_data` and `feature_cache`.

- 0002_add_unique_prediction_outcomes_prediction_id
  - Adds `UNIQUE(prediction_id)` on `prediction_outcomes` if duplicates do not exist.
  - If duplicates are found, the migration logs a NOTICE and skips adding the constraint. Use the fixer script below and re-run.

- 0003_futures_table_and_hypertable
  - Formalizes `futures_data` table (matches runtime usage).
  - Adds index on `(symbol, timestamp DESC)` and `UNIQUE(symbol, timestamp)`.
  - Best-effort TimescaleDB hypertable conversion for `futures_data` with data migration when necessary.

## Handling known constraint issue (prediction_outcomes)

If the 0002 migration logs a NOTICE about duplicates:
1. Run the fixer:
   - `python scripts/fix_prediction_outcomes_constraint.py` (inside container or locally with DB access)
2. Re-run the migration:
   - `make db-upgrade` (or restart the app so it applies on startup)

This ensures the unique constraint can be applied safely.

## TimescaleDB considerations

- The dev environment uses `timescale/timescaledb` image and mounts `sql/init.sql`, which calls `CREATE EXTENSION IF NOT EXISTS timescaledb;`.
- In other environments, creating extensions might require elevated privileges. Migrations use best-effort blocks:
  - If extension creation is not permitted, migrations continue without failing.
  - Hypertable creation is attempted only if TimescaleDB is available.
- This approach avoids breaking app startup due to missing superuser permissions.

## Interaction with sql/init.sql

- Dev Docker still mounts `sql/init.sql` for first boot of a new volume.
- Migrations are designed to be idempotent and skip creation when objects already exist (using `IF NOT EXISTS` / guarded `DO` blocks).
- Over time, we can rely solely on Alembic for greenfield creation and drop the mount, once verified across environments.

## Best practices for writing migrations

- Favor idempotent DDL:
  - Use `CREATE TABLE IF NOT EXISTS`, `CREATE INDEX IF NOT EXISTS`, `CREATE MATERIALIZED VIEW IF NOT EXISTS`.
  - For constraints or operations without `IF NOT EXISTS`, guard with checks via `information_schema` or wrap in PL/pgSQL `DO $$ ... $$;` blocks.
- Consider existing data:
  - When adding unique constraints, pre-check for duplicates.
  - Provide remediation steps (scripts) and make migrations non-fatal when conflicts are found.
- TimescaleDB:
  - Wrap hypertable creation in try/catch blocks to avoid breaking environments without the extension.

## Deprecation note: code-based futures table creation

- The project previously created `futures_data` on-demand in runtime code (`DatabaseOperations.create_futures_table()`), and `src/main.py` invoked it on startup.
- Migration `0003` formalizes this table.
- We will keep the code path for now to avoid surprises in non-standard environments.
- Once comfortable with the migration-based flow, we can remove the on-demand creation:
  - Remove the call to `create_futures_table()` in `src/main.py`.
  - Optionally delete the method to prevent schema mutations outside Alembic.

## CI/CD considerations

- Recommended: apply migrations as a deployment step prior to starting the app (e.g., `alembic upgrade head`).
- The app already self-applies migrations on startup; this is sufficient for many setups, but explicit migration steps in CI/CD give clearer control and logs.

## Troubleshooting

- “Insufficient privilege to create extension timescaledb”: This is expected in restricted environments. Migrations continue; hypertables will be skipped.
- “Duplicate rows prevent unique constraint”: Run `scripts/fix_prediction_outcomes_constraint.py` and re-apply migrations.
- “Object already exists” errors: Use `IF NOT EXISTS` semantics or guard with `information_schema` checks in new migrations.

## Useful commands

- Show current containers:
  - `make status`
- Start dev environment:
  - `make run`
- Check health:
  - `make health`
- Apply migrations:
  - `make db-upgrade`
- Create new revision:
  - `make db-revision NAME="short message"`
