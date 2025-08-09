.PHONY: help prepare run stop logs health test clean deploy build rebuild predictions predictions-accuracy predictions-pending predictions-stats predictions-today predictions-highconf data-integrity env restart status predict backup models-history models-clear models-list dashboard dashboard-logs dashboard-build nuke deploy db-revision db-upgrade tune

# Auto-detect environment from .env file, allow override with ENV variable
DETECTED_ENV := $(shell grep -E '^ENVIRONMENT=' .env 2>/dev/null | cut -d'=' -f2 || echo "development")
CURRENT_ENV := $(or $(ENV),$(DETECTED_ENV))

# Set compose file based on environment
ifeq ($(CURRENT_ENV),production)
    COMPOSE_FILE := docker-compose.prod.yml
    APP_CONTAINER := bse-predictor-prod
    DB_CONTAINER := bse-postgres-prod
else
    COMPOSE_FILE := docker-compose.dev.yml
    APP_CONTAINER := bse-predict-app-dev
    DB_CONTAINER := bse-predict-postgres-dev
endif

# Default target
help:
	@echo "BSE Predict - Makefile Commands"
	@echo "=============================="
	@echo ""
	@echo "Tuning (Hyperparameter Optimization):"
	@echo "  make tune SYMBOL=\"BTC/USDT\" TARGET=0.02 MODEL=rf [TRIALS=50]"
	@echo "      Runs Optuna HPO storing trials and results in Postgres."
	@echo "      MODEL options: rf | xgb | lgb | gb"
	@echo ""
	@echo "Main Commands:"
	@echo "  make prepare    - Prepare system for first run"
	@echo "  make run        - Run environment (auto-detects from .env)"
	@echo "  make stop       - Stop all containers"
	@echo "  make logs       - Follow application logs"
	@echo "  make health     - Check system health"
	@echo "  make restart    - Restart application container"
	@echo ""
	@echo "Testing:"
	@echo "  make test       - Run all tests"
	@echo "  make test-quick - Run quick integration tests"
	@echo "  make test-phase4 - Run Phase 4 tests only"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean      - Clean up containers and volumes"
	@echo "  make build      - Build Docker images"
	@echo "  make rebuild    - Rebuild images from scratch (no cache)"
	@echo "  make restart    - Restart application container"
	@echo "  make backup     - Backup database"
	@echo "  make data-integrity - Check data integrity and statistics"
	@echo "  make errors     - Search for errors in container logs"
	@echo ""
	@echo "Environment:"
	@echo "  Current: $(CURRENT_ENV) (detected from .env)"
	@echo "  All commands automatically use the environment from .env file"
	@echo "  To override: make <command> ENV=production"
	@echo "  To switch: edit ENVIRONMENT= in .env file"
	@echo ""
	@echo "Predictions:"
	@echo "  make predictions         - Show recent predictions"
	@echo "  make predictions-accuracy - Show accuracy summary"
	@echo "  make predictions-pending  - Show pending predictions"
	@echo "  make predictions-stats    - Show overall statistics"
	@echo "  make check-outcomes      - Check and update prediction outcomes"
	@echo ""
	@echo "Models:"
	@echo "  make models-history      - Show model training history"
	@echo "  make models-clear        - Clear all models (local and Docker)"
	@echo "  make models-list         - List current models and their paths"
	@echo ""
	@echo "Dashboard:"
	@echo "  make dashboard           - Open dashboard in browser"
	@echo "  make dashboard-logs      - View dashboard logs"
	@echo "  make dashboard-build     - Build dashboard images"
	@echo ""
	@echo "Migrations:"
	@echo "  make db-revision NAME=\"message\"   - Create a new Alembic revision (inside container if running, else locally)"
	@echo "  make db-upgrade [REV=head]         - Apply Alembic migrations to target revision (default: head)"
	@echo ""

# Prepare system for first run
prepare:
	@./scripts/prepare_first_run.sh

# Run environment (auto-detects from .env)
run:
	@echo "🚀 Starting $(CURRENT_ENV) environment..."
	@docker-compose -f $(COMPOSE_FILE) up -d
	@echo "✅ $(CURRENT_ENV) environment started"
	@if [ "$(CURRENT_ENV)" = "development" ]; then \
		echo "📱 App: http://localhost:8000/health"; \
		echo "📊 Dashboard: http://localhost:3000"; \
		echo "🔌 Dashboard API: http://localhost:8001"; \
		echo "🗄️ pgAdmin: http://localhost:8080"; \
	else \
		echo "📊 Dashboard: http://localhost:3000"; \
		echo "🔌 Dashboard API: http://localhost:8001"; \
	fi
	@echo "📊 Logs: make logs"

# Stop all containers
stop:
	@echo "🛑 Stopping containers ($(CURRENT_ENV))..."
	@docker-compose -f $(COMPOSE_FILE) down
	@echo "✅ Containers stopped"

# Follow logs
logs:
	@docker-compose -f $(COMPOSE_FILE) logs -f

# Check system health
health:
	@echo "💓 Checking system health..."
	@curl -s http://localhost:8000/health | jq . || echo "❌ Health check failed - is the system running?"

# Run all tests
test:
	@echo "🧪 Running all tests..."
	@./scripts/test_with_docker.sh

# Quick integration test
test-quick:
	@echo "🚀 Running quick tests..."
	@source venv/bin/activate && python tests/test_integrated.py --quick

# Phase 4 specific tests
test-phase4:
	@echo "🐳 Running Phase 4 tests..."
	@source venv/bin/activate && python tests/test_integrated.py --phase "Phase 4"

# Build Docker images
build:
	@echo "🔨 Building Docker images ($(CURRENT_ENV))..."
	@docker-compose -f $(COMPOSE_FILE) build

# Rebuild from scratch (no cache)
rebuild:
	@echo "🔄 Rebuilding Docker images from scratch ($(CURRENT_ENV))..."
	@docker-compose -f $(COMPOSE_FILE) build --no-cache


# Clean up everything
clean:
	@echo "🧹 Cleaning up ($(CURRENT_ENV))..."
	@docker-compose -f $(COMPOSE_FILE) down -v
	@echo "✅ Cleanup complete"

# Restart application
restart:
	@echo "🔄 Restarting application ($(CURRENT_ENV))..."
	@docker-compose -f $(COMPOSE_FILE) restart $(APP_CONTAINER)
	@echo "✅ Application restarted"

# Database commands
db-up:
	@echo "🗄️ Starting database ($(CURRENT_ENV))..."
	@docker-compose -f $(COMPOSE_FILE) up -d postgres
	@echo "✅ Database started"

db-down:
	@echo "🗄️ Stopping database ($(CURRENT_ENV))..."
	@docker-compose -f $(COMPOSE_FILE) stop postgres
	@echo "✅ Database stopped"

# Show container status
status:
	@echo "📊 Container Status ($(CURRENT_ENV)):"
	@docker-compose -f $(COMPOSE_FILE) ps
	@echo ""
	@echo "📈 Resource Usage:"
	@docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# Run manual prediction
predict:
	@echo "🔮 Running manual prediction ($(CURRENT_ENV))..."
	@echo "   Container: $(APP_CONTAINER)"
	@docker exec $(APP_CONTAINER) python -c "from src.scheduler.task_scheduler import MultiTargetTaskScheduler; from src.database.operations import db_ops; from src.utils.config import config; s = MultiTargetTaskScheduler(db_ops, config); s.initialize_components(); s.run_manual_prediction()"


# Backup database
backup:
	@echo "💾 Backing up database ($(CURRENT_ENV))..."
	@mkdir -p backups
	@docker exec $(DB_CONTAINER) pg_dump -U crypto_user crypto_ml | gzip > backups/backup_$(CURRENT_ENV)_$$(date +%Y%m%d_%H%M%S).sql.gz
	@echo "✅ Backup saved to backups/"

# Show recent predictions
predictions:
	@./scripts/check_predictions.sh recent

# Show accuracy summary
predictions-accuracy:
	@./scripts/check_predictions.sh accuracy

# Show pending predictions
predictions-pending:
	@./scripts/check_predictions.sh pending

# Show overall statistics
predictions-stats:
	@./scripts/check_predictions.sh stats

# Show today's performance
predictions-today:
	@./scripts/check_predictions.sh today

# Show high confidence winners
predictions-highconf:
	@./scripts/check_predictions.sh highconf

# Check and update prediction outcomes
# Usage: make check-outcomes [HOURS=72]
check-outcomes:
	@./scripts/check_outcomes.sh $(or $(HOURS),72)

# Check data integrity and statistics
data-integrity:
	@./scripts/check_data_integrity.sh

# Search for errors in logs
errors:
	@echo "🔍 Searching for errors in Docker container logs..."
	@echo ""
	@echo "📋 Error Summary (Recent Container Logs):"
	@echo "──────────────────────────────────────────"
	@if docker ps -q -f name=$(APP_CONTAINER) >/dev/null 2>&1; then \
		LOGS=$$(docker logs $(APP_CONTAINER) 2>&1 --tail 10000); \
		echo "Total ERROR lines: $$(echo "$$LOGS" | grep -c 'ERROR' 2>/dev/null || echo 0)"; \
		echo "Total WARNING lines: $$(echo "$$LOGS" | grep -c 'WARNING' 2>/dev/null || echo 0)"; \
		echo "Total CRITICAL lines: $$(echo "$$LOGS" | grep -c 'CRITICAL' 2>/dev/null || echo 0)"; \
		echo "Total Exception traces: $$(echo "$$LOGS" | grep -c 'Traceback' 2>/dev/null || echo 0)"; \
	else \
		echo "❌ Container $(APP_CONTAINER) is not running"; \
	fi
	@echo ""
	@echo "🚨 Last 10 ERROR messages:"
	@echo "──────────────────────────"
	@if docker ps -q -f name=$(APP_CONTAINER) >/dev/null 2>&1; then \
		docker logs $(APP_CONTAINER) 2>&1 --tail 5000 | grep 'ERROR' | tail -10 || echo "No errors found"; \
	fi
	@echo ""
	@echo "⚠️  Last 5 WARNING messages:"
	@echo "────────────────────────────"
	@if docker ps -q -f name=$(APP_CONTAINER) >/dev/null 2>&1; then \
		docker logs $(APP_CONTAINER) 2>&1 --tail 5000 | grep 'WARNING' | tail -5 || echo "No warnings found"; \
	fi
	@echo ""
	@echo "💀 Last 5 CRITICAL messages:"
	@echo "─────────────────────────────"
	@if docker ps -q -f name=$(APP_CONTAINER) >/dev/null 2>&1; then \
		docker logs $(APP_CONTAINER) 2>&1 --tail 5000 | grep 'CRITICAL' | tail -5 || echo "No critical errors found"; \
	fi
	@echo ""
	@echo "🐛 Last Exception (with context):"
	@echo "──────────────────────────────────"
	@if docker ps -q -f name=$(APP_CONTAINER) >/dev/null 2>&1; then \
		docker logs $(APP_CONTAINER) 2>&1 --tail 5000 | grep -A 10 'Traceback' | tail -15 || echo "No exceptions found"; \
	fi
	@echo ""
	@echo "📊 Error frequency by module (if available):"
	@echo "─────────────────────────────────────────────"
	@if docker ps -q -f name=$(APP_CONTAINER) >/dev/null 2>&1; then \
		docker logs $(APP_CONTAINER) 2>&1 --tail 5000 | grep 'ERROR' | grep -oE '\[[\w\.]+\]' | sort | uniq -c | sort -rn | head -10 || echo "No module pattern found"; \
	fi
	@echo ""
	@echo "💡 Tip: Use 'make logs' to see live container logs"
	@echo "💡 Tip: Check logs inside container: docker exec $(APP_CONTAINER) cat /app/logs/app.log"

# Show current environment
env:
	@echo "🌍 Environment Configuration:"
	@echo "  Detected from .env: $(DETECTED_ENV)"
	@echo "  Current environment: $(CURRENT_ENV)"
	@echo "  Compose file: $(COMPOSE_FILE)"
	@echo "  App container: $(APP_CONTAINER)"
	@echo "  DB container: $(DB_CONTAINER)"

# Show model training history
models-history:
	@echo "📊 Model Training History ($(CURRENT_ENV))..."
	@docker exec $(APP_CONTAINER) python -c "from src.models.multi_target_trainer import MultiTargetModelTrainer; trainer = MultiTargetModelTrainer(); df = trainer.get_training_history(20); print('\n🎯 Recent Model Training History\n' if not df.empty else 'No training history found'); print(df[['symbol', 'target_pct', 'trained_at', 'training_samples', 'cv_score', 'final_accuracy', 'precision', 'recall', 'f1_score']].to_string(index=False)) if not df.empty else None"

# Clear all models (local and Docker)
models-clear:
	@echo "🗑️ Clearing all models..."
	@echo "⚠️  This will delete all trained models!"
	@echo -n "Are you sure? (y/N) "; \
	read REPLY; \
	if [ "$$REPLY" = "y" ] || [ "$$REPLY" = "Y" ]; then \
		echo "🧹 Clearing local models..."; \
		rm -f models/*.pkl 2>/dev/null || true; \
		echo "✅ Local models cleared"; \
		if [ -n "$$(docker ps -q -f name=$(APP_CONTAINER) 2>/dev/null)" ]; then \
			echo "🐳 Clearing Docker models..."; \
			docker exec $(APP_CONTAINER) sh -c "rm -f /app/models/*.pkl 2>/dev/null || true"; \
			echo "✅ Docker models cleared"; \
		else \
			echo "🐳 Docker container not running"; \
		fi; \
		echo ""; \
		echo "💡 Models will be trained automatically on next startup"; \
	else \
		echo "❌ Cancelled"; \
	fi

# List current models
models-list:
	@echo "📂 Current Models ($(CURRENT_ENV))..."
	@echo ""
	@echo "🏠 Local models:"
	@if [ -d models ] && [ "$$(ls -A models/*.pkl 2>/dev/null)" ]; then \
		ls -la models/*.pkl | awk '{print "   " $$9 " (" $$5 " bytes)"}'; \
	else \
		echo "   No local models found"; \
	fi
	@echo ""
	@if [ -n "$$(docker ps -q -f name=$(APP_CONTAINER))" ]; then \
		echo "🐳 Docker models:"; \
		docker exec $(APP_CONTAINER) sh -c 'if [ "$$(ls -A /app/models/*.pkl 2>/dev/null)" ]; then ls -la /app/models/*.pkl | awk "{print \"   \" \$$9 \" (\" \$$5 \" bytes)\"}"; else echo "   No Docker models found"; fi'; \
	else \
		echo "🐳 Docker container not running"; \
	fi

# Dashboard commands
dashboard:
	@echo "📊 Opening BSE Predict Dashboard..."
	@if [ "$(CURRENT_ENV)" = "development" ]; then \
		echo "Dashboard: http://localhost:3000"; \
		echo "API Docs: http://localhost:8001/docs"; \
	else \
		echo "Dashboard: http://localhost:3000"; \
	fi
	@command -v open >/dev/null 2>&1 && open http://localhost:3000 || \
		command -v xdg-open >/dev/null 2>&1 && xdg-open http://localhost:3000 || \
		echo "Please open http://localhost:3000 in your browser"

# Dashboard logs
dashboard-logs:
	@echo "📊 Dashboard logs..."
	@docker-compose -f $(COMPOSE_FILE) logs -f dashboard-backend dashboard-frontend

# Build dashboard images
dashboard-build:
	@echo "🔨 Building dashboard images..."
	@docker-compose -f $(COMPOSE_FILE) build dashboard-backend dashboard-frontend
	@echo "✅ Dashboard images built"

# Deploy: pull latest changes and restart
deploy:
	@echo "🚀 Deploying latest changes..."
	@echo "1️⃣  Pulling latest changes from git..."
	@git pull
	@echo ""
	@echo "2️⃣  Stopping containers..."
	@make stop
	@echo ""
	@echo "3️⃣  Building updated images..."
	@make build
	@echo ""
	@echo "4️⃣  Starting services..."
	@make run
	@echo ""
	@echo "✅ Deployment complete!"

# Nuclear option - delete everything and rebuild from scratch
nuke:
	@echo "☢️  NUCLEAR OPTION - Complete System Reset"
	@echo "⚠️  This will DELETE:"
	@echo "  • All containers"
	@echo "  • All volumes (database data)"
	@echo "  • All images"
	@echo "  • All build cache"
	@echo ""
	@echo -n "Are you ABSOLUTELY sure? Type 'yes' to confirm: "; \
	read REPLY; \
	if [ "$$REPLY" = "yes" ]; then \
		echo ""; \
		echo "💥 Initiating nuclear cleanup..."; \
		echo ""; \
		echo "1️⃣  Stopping all containers..."; \
		docker-compose -f $(COMPOSE_FILE) down -v; \
		echo ""; \
		echo "2️⃣  Removing volumes..."; \
		docker volume rm bse-predict_postgres_data bse-predict_model_data 2>/dev/null || true; \
		echo ""; \
		echo "3️⃣  Removing containers..."; \
		docker-compose -f $(COMPOSE_FILE) rm -f; \
		echo ""; \
		echo "4️⃣  Removing images..."; \
		docker rmi bse-predict-dashboard-backend bse-predict-dashboard-frontend bse-predictor 2>/dev/null || true; \
		echo ""; \
		echo "5️⃣  Cleaning build cache..."; \
		docker builder prune -f; \
		echo ""; \
		echo "6️⃣  Rebuilding everything from scratch..."; \
		docker-compose -f $(COMPOSE_FILE) build --no-cache; \
		echo ""; \
		echo "7️⃣  Starting fresh system..."; \
		docker-compose -f $(COMPOSE_FILE) up -d; \
		echo ""; \
		echo "✅ Nuclear cleanup complete! System rebuilt from scratch."; \
		echo ""; \
		echo "📋 Next steps:"; \
		echo "  1. Wait ~30 seconds for database to initialize"; \
		echo "  2. Run 'make health' to check system status"; \
		echo "  3. Dashboard will be available at http://localhost:3000"; \
	else \
		echo "❌ Cancelled - no changes made"; \
	fi

# Alembic migration helpers
db-revision:
	@if [ -z "$(NAME)" ]; then echo "❌ ERROR: Provide NAME=\"message\". Example: make db-revision NAME=\"add new table\""; exit 1; fi
	@echo "📦 Creating Alembic revision: $(NAME)"
	@if docker ps -q -f name=$(APP_CONTAINER) >/dev/null 2>&1; then \
		echo "🐳 Using running container: $(APP_CONTAINER)"; \
		docker exec -w /app $(APP_CONTAINER) bash -lc 'python -m alembic revision -m "$$NAME"' NAME="$(NAME)"; \
	else \
		echo "🖥️  No app container running, trying local environment..."; \
		python -m alembic revision -m "$(NAME)"; \
	fi
	@echo "✅ Revision created under src/database/migrations/versions"

db-upgrade:
	@echo "🚀 Applying Alembic migrations (target: $(or $(REV),head))"
	@if docker ps -q -f name=$(APP_CONTAINER) >/dev/null 2>&1; then \
		echo "🐳 Using running container: $(APP_CONTAINER)"; \
		docker exec -w /app $(APP_CONTAINER) bash -lc 'python -m alembic upgrade "$${REV:-head}"' REV="$(REV)"; \
	else \
		echo "🖥️  No app container running, trying local environment..."; \
		python -m alembic upgrade $(or $(REV),head); \
	fi
	@echo "✅ Migrations applied"

# Hyperparameter Optimization (Optuna)
# Usage examples:
#   make tune SYMBOL="BTC/USDT" TARGET=0.02 MODEL=rf TRIALS=50
#   make tune SYMBOL="ETH/USDT" TARGET=0.01 MODEL=xgb
tune:
	@if [ -z "$(SYMBOL)" ] || [ -z "$(TARGET)" ] || [ -z "$(MODEL)" ]; then \
		echo "❌ Missing args. Usage: make tune SYMBOL=\"BTC/USDT\" TARGET=0.02 MODEL=rf [TRIALS=50]"; \
		exit 1; \
	fi; \
	echo "🔧 Running HPO: SYMBOL=$(SYMBOL) TARGET=$(TARGET) MODEL=$(MODEL) TRIALS=$(TRIALS)"; \
	if [ -n "$$(docker ps -q -f name=$(APP_CONTAINER) 2>/dev/null)" ]; then \
		echo "🐳 Executing inside container: $(APP_CONTAINER)"; \
		if [ -n "$(TRIALS)" ]; then \
			docker exec $(APP_CONTAINER) python -m src.tuning.optuna_tuner --symbol "$(SYMBOL)" --target $(TARGET) --model $(MODEL) --trials $(TRIALS); \
		else \
			docker exec $(APP_CONTAINER) python -m src.tuning.optuna_tuner --symbol "$(SYMBOL)" --target $(TARGET) --model $(MODEL); \
		fi; \
	else \
		echo "🖥️  Executing locally"; \
		if [ -n "$(TRIALS)" ]; then \
			python -m src.tuning.optuna_tuner --symbol "$(SYMBOL)" --target $(TARGET) --model $(MODEL) --trials $(TRIALS); \
		else \
			python -m src.tuning.optuna_tuner --symbol "$(SYMBOL)" --target $(TARGET) --model $(MODEL); \
		fi; \
	fi
