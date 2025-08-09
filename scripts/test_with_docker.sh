#!/bin/bash
# Run tests with Docker database

set -e

echo "🧪 Running tests with Docker database"
echo "===================================="

# Detect environment from .env file
ENV_FILE=".env"
if [ -f "$ENV_FILE" ]; then
    ENVIRONMENT=$(grep -E "^ENVIRONMENT=" "$ENV_FILE" | cut -d'=' -f2)
else
    ENVIRONMENT="development"
fi

# Set container name based on environment
if [ "$ENVIRONMENT" = "production" ]; then
    DB_CONTAINER="bse-postgres-prod"
else
    DB_CONTAINER="bse-predict-postgres-dev"
fi

# Check if database is running
if ! docker ps | grep -q $DB_CONTAINER; then
    echo "❌ Database container ($DB_CONTAINER) is not running!"
    echo "Please run 'make run' first"
    exit 1
fi

# Export test environment variables
export DATABASE_URL="postgresql://crypto_user:password@localhost:5432/crypto_ml"
export TELEGRAM_BOT_TOKEN="test_token"
export TELEGRAM_CHAT_ID="test_chat"

# Run tests
echo ""
echo "Running integrated tests..."
source venv/bin/activate
python tests/test_integrated.py