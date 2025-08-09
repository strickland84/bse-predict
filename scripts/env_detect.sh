#!/bin/bash
# Environment detection script - source this from other scripts
# Usage: source scripts/env_detect.sh

# Detect environment from .env file
ENV_FILE=".env"
if [ -f "$ENV_FILE" ]; then
    ENVIRONMENT=$(grep -E "^ENVIRONMENT=" "$ENV_FILE" | cut -d'=' -f2)
    # Also load POSTGRES_PASSWORD for backup operations
    POSTGRES_PASSWORD=$(grep -E "^POSTGRES_PASSWORD=" "$ENV_FILE" | cut -d'=' -f2)
else
    ENVIRONMENT="development"
    POSTGRES_PASSWORD="password"
fi

# Set container names based on environment
if [ "$ENVIRONMENT" = "production" ]; then
    DB_CONTAINER="bse-postgres-prod"
    APP_CONTAINER="bse-predictor-prod"
    COMPOSE_FILE="docker-compose.prod.yml"
else
    DB_CONTAINER="bse-predict-postgres-dev"
    APP_CONTAINER="bse-predict-app-dev"
    COMPOSE_FILE="docker-compose.dev.yml"
fi

# Database settings
DB_NAME="crypto_ml"
DB_USER="crypto_user"

# Export for use in scripts
export ENVIRONMENT
export DB_CONTAINER
export APP_CONTAINER
export COMPOSE_FILE
export DB_NAME
export DB_USER
export POSTGRES_PASSWORD