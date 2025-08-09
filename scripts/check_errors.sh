#!/bin/bash

# Check system errors from database

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Detect environment from .env file
ENV_FILE=".env"
if [ -f "$ENV_FILE" ]; then
    ENVIRONMENT=$(grep -E "^ENVIRONMENT=" "$ENV_FILE" | cut -d'=' -f2)
else
    ENVIRONMENT="development"
fi

# Set container names based on environment
if [ "$ENVIRONMENT" = "production" ]; then
    DB_CONTAINER="bse-postgres-prod"
else
    DB_CONTAINER="bse-predict-postgres-dev"
fi

# Database settings
DB_NAME="crypto_ml"
DB_USER="crypto_user"

# Always use docker exec for both environments
DB_CMD="docker exec -i $DB_CONTAINER psql -U $DB_USER -d $DB_NAME"

echo -e "${BLUE}📋 System Error Log Analysis${NC}"
echo -e "${BLUE}================================${NC}"
echo

# Check if database is accessible
if ! echo "SELECT 1" | $DB_CMD -t >/dev/null 2>&1; then
    echo -e "${RED}❌ Cannot connect to database${NC}"
    exit 1
fi

# Function to run queries
run_query() {
    local query="$1"
    echo "$query" | $DB_CMD -t 2>/dev/null
}

# 1. Error summary by level (last 24 hours)
echo -e "${YELLOW}📊 Error Summary (Last 24 Hours)${NC}"
query="
SELECT 
    level,
    COUNT(*) as count,
    COUNT(DISTINCT logger_name) as unique_loggers,
    COUNT(DISTINCT module) as unique_modules
FROM system_errors
WHERE timestamp >= NOW() - INTERVAL '24 hours'
GROUP BY level
ORDER BY 
    CASE level
        WHEN 'CRITICAL' THEN 1
        WHEN 'ERROR' THEN 2
        WHEN 'WARNING' THEN 3
        ELSE 4
    END;
"
run_query "$query"
echo

# 2. Error trends by hour
echo -e "${YELLOW}📈 Error Trends (Hourly)${NC}"
query="
SELECT 
    DATE_TRUNC('hour', timestamp) as hour,
    COUNT(*) as errors
FROM system_errors
WHERE timestamp >= NOW() - INTERVAL '24 hours'
GROUP BY DATE_TRUNC('hour', timestamp)
ORDER BY hour DESC
LIMIT 10;
"
run_query "$query"
echo

# 3. Top error sources
echo -e "${YELLOW}🔍 Top Error Sources${NC}"
query="
SELECT 
    logger_name,
    module,
    COUNT(*) as error_count
FROM system_errors
WHERE timestamp >= NOW() - INTERVAL '24 hours'
GROUP BY logger_name, module
ORDER BY error_count DESC
LIMIT 10;
"
run_query "$query"
echo

# 4. Recent errors
echo -e "${YELLOW}🚨 Recent Errors (Last 10)${NC}"
query="
SELECT 
    TO_CHAR(timestamp, 'MM-DD HH24:MI') as time,
    level,
    SUBSTRING(logger_name, 1, 30) as logger,
    SUBSTRING(message, 1, 80) as message
FROM system_errors
ORDER BY timestamp DESC
LIMIT 10;
"
run_query "$query"
echo

# 5. Critical errors with details
echo -e "${YELLOW}💀 Critical Errors (Last 7 Days)${NC}"
query="
SELECT 
    TO_CHAR(timestamp, 'YYYY-MM-DD HH24:MI') as time,
    logger_name,
    SUBSTRING(message, 1, 100) as message,
    CASE 
        WHEN exception IS NOT NULL THEN '✓'
        ELSE '✗'
    END as has_exception
FROM system_errors
WHERE level = 'CRITICAL'
    AND timestamp >= NOW() - INTERVAL '7 days'
ORDER BY timestamp DESC
LIMIT 10;
"
result=$(run_query "$query")
if [ -z "$result" ] || [ "$result" = "(0 rows)" ]; then
    echo -e "${GREEN}✅ No critical errors in the last 7 days${NC}"
else
    echo "$result"
fi
echo

# 6. Errors with exceptions
echo -e "${YELLOW}🐛 Errors with Exceptions (Last 24h)${NC}"
query="
SELECT 
    TO_CHAR(timestamp, 'MM-DD HH24:MI') as time,
    module || '.' || function || ':' || line_number as location,
    SUBSTRING(exception, 1, 80) as exception
FROM system_errors
WHERE exception IS NOT NULL
    AND timestamp >= NOW() - INTERVAL '24 hours'
ORDER BY timestamp DESC
LIMIT 5;
"
run_query "$query"
echo

# 7. Database size for errors table
echo -e "${YELLOW}💾 Error Log Storage${NC}"
query="
SELECT 
    COUNT(*) as total_errors,
    pg_size_pretty(pg_relation_size('system_errors')) as table_size,
    MIN(timestamp) as oldest_error,
    MAX(timestamp) as newest_error
FROM system_errors;
"
run_query "$query"

# Check if table exists
query="
SELECT EXISTS (
    SELECT FROM information_schema.tables 
    WHERE table_schema = 'public' 
    AND table_name = 'system_errors'
) as table_exists;
"
exists=$(run_query "$query" | grep -o 't\|f' | head -1)

if [ "$exists" = "f" ]; then
    echo
    echo -e "${RED}❌ system_errors table does not exist${NC}"
    echo -e "${YELLOW}Run 'make init-db' to create it${NC}"
fi