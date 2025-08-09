# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BSE Predict is a multi-target cryptocurrency price prediction system that uses machine learning to predict ±1%, ±2%, and ±5% price movements for BTC, ETH, and SOL. The system provides hourly predictions with confidence scores and delivers alerts via Telegram.

## Key Architecture Components

1. **Data Pipeline**: CCXT → PostgreSQL/TimescaleDB → Feature Engineering → ML Models
2. **ML Stack**: Multiple RandomForest models (9 total: 3 assets × 3 targets) with time-series cross-validation
3. **Notification System**: Telegram bot for hourly reports and high-confidence alerts
4. **Scheduler**: Automated tasks for data fetching, predictions, model retraining

## Development Commands

### Environment Setup
```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Database Operations
```bash
# Start development database
docker-compose -f docker-compose.dev.yml up -d

# Apply Alembic migrations to latest (the app also runs migrations automatically on startup)
make db-upgrade

# Or programmatically (uses the in-repo runner)
python - <<'PY'
import os
from src.database.migrations_runner import run_migrations
run_migrations(os.environ.get("DATABASE_URL", "postgresql://crypto_user:dev_password@localhost:5432/crypto_ml"))
PY

# Test database connection
python -c "from src.utils.config import Config; from src.database.connection import DatabaseConnection; print(DatabaseConnection(Config().database_url).test_connection())"
```

### Migrations (Alembic)
```bash
# Create a new migration
make db-revision NAME="add new table"

# Upgrade to latest (or a specific revision with REV=<revision_id>)
make db-upgrade
make db-upgrade REV=1a2b3c4d5e6f
```

Key locations:
- alembic.ini (root)
- src/database/migrations/env.py
- src/database/migrations/versions/
- src/database/migrations_runner.py
- src/database/init_db.py (runs migrations on app startup)

### Testing
```bash
# Run integrated test suite (recommended)
python tests/test_integrated.py              # Full system validation
python tests/test_integrated.py --quick      # Quick validation
python tests/test_integrated.py --phase "Phase 2"  # Specific phase

# Run individual phase tests
python tests/test_phase1.py  # Foundation tests (9/9 passing)
python tests/test_phase2.py  # ML model tests (6/6 passing)
python tests/test_phase3.py  # Telegram integration tests (5/5 passing)
```

### Data Operations
```bash
# Fetch latest data for all assets
python -c "from src.data.fetcher import ExchangeDataFetcher; f = ExchangeDataFetcher(); print(f.fetch_latest_candle('BTC/USDT'))"

# Check data health and recover missing data
python -c "from src.data.recovery import recovery_manager; print(recovery_manager.get_data_health_report())"

# Ensure minimum data coverage for all symbols
python -c "from src.data.recovery import recovery_manager; recovery_manager.ensure_all_symbols_coverage()"
```

### Model Training & Prediction
```bash
# Train all models
python -c "from src.models.multi_target_trainer import MultiTargetModelTrainer; t = MultiTargetModelTrainer(); t.train_all_models()"

# Make predictions for all assets
python -c "from src.models.multi_target_predictor import MultiTargetPredictionEngine; p = MultiTargetPredictionEngine(); p.predict_all_assets(['BTC/USDT', 'ETH/USDT', 'SOL/USDT'])"

# Run manual prediction with Telegram report
python -c "from src.scheduler.task_scheduler import MultiTargetTaskScheduler; from src.database.operations import db_ops; from src.utils.config import config; s = MultiTargetTaskScheduler(db_ops, config); s.initialize_components(); s.run_manual_prediction()"
```

### Code Quality
```bash
# Format code with black
black src/ tests/

# Check code style
flake8 src/ tests/

# Type checking
mypy src/
```

## Code Architecture

### Core Modules

- `src/data/`: Data fetching and feature engineering
  - `fetcher.py`: CCXT-based exchange data fetching
  - `feature_engineer.py`: Technical indicators and multi-target label creation
  - `recovery.py`: Data gap detection and recovery

- `src/models/`: ML models for price prediction
  - `multi_target_trainer.py`: Trains 9 separate models (3 assets × 3 targets)
  - `multi_target_predictor.py`: Makes predictions using trained models

- `src/database/`: PostgreSQL/TimescaleDB operations
  - `connection.py`: Database connection management
  - `operations.py`: CRUD operations for OHLCV data

- `src/notifications/`: Telegram integration
  - `telegram_reporter.py`: Sends hourly reports and high-confidence alerts

- `src/scheduler/`: Task automation
  - `task_scheduler.py`: Manages data fetching, predictions, retraining

### Key Design Patterns

1. **Singleton Pattern**: Database connection (`db_connection`) and operations (`db_ops`)
2. **Factory Pattern**: Model creation based on target percentage in `MultiTargetModelTrainer`
3. **Strategy Pattern**: Different model configurations for 1%, 2%, 5% targets
4. **Observer Pattern**: Telegram alerts for high-confidence predictions

### Important Configuration

- `config.yaml`: Main configuration file (database, telegram, trading parameters)
- Environment variables required:
  - `DATABASE_URL`: PostgreSQL connection string
  - `TELEGRAM_BOT_TOKEN`: Bot authentication token
  - `TELEGRAM_CHAT_ID`: Target chat for notifications

### Model Training Strategy

- Each target percentage (1%, 2%, 5%) uses different RandomForest parameters
- Time-series cross-validation prevents future data leakage
- Models are retrained daily with fresh data
- Feature importance tracking for model interpretability

### Error Handling

- All database operations have retry logic
- Exchange API calls use rate limiting
- Telegram notifications fail gracefully
- Scheduler continues running even if individual tasks fail

## Production Deployment

The system is designed for VPS deployment with:
- Docker containers for PostgreSQL/TimescaleDB
- Python application running with systemd
- Automated backups and monitoring
- ~€8-12/month operational costs

## Phase Implementation Status

- **Phase 1** (✅ Complete): Foundation & Data Pipeline - All tests passing
- **Phase 2** (✅ Complete): ML Prediction Engine - All tests passing
- **Phase 3** (✅ Complete): Telegram Integration - All tests passing
- **Phase 4** (🔄 Next): Scheduler & Automation - Tests ready, implementation pending
- **Phase 5** (📋 Planned): Production Deployment - Tests ready, implementation pending

## Testing Approach

The project uses a comprehensive integrated testing suite (`tests/test_integrated.py`) that:
- Tests all phases in sequence
- Validates system integration
- Provides quick validation mode
- Allows phase-specific testing
- Is easily expandable for future phases
