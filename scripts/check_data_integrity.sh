#!/bin/bash
# Script to check data integrity and show data statistics

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
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
run_query() {
    docker exec $DB_CONTAINER psql -U $DB_USER -d $DB_NAME -t -A -F"," -c "$1" 2>/dev/null
}

run_pretty_query() {
    docker exec $DB_CONTAINER psql -U $DB_USER -d $DB_NAME -c "$1"
}

echo -e "${BLUE}=== BSE PREDICT DATA INTEGRITY CHECK ===${NC}"
echo -e "${BLUE}$(date '+%Y-%m-%d %H:%M:%S UTC')${NC}"
echo "========================================"

# Check OHLCV data
echo -e "\n${CYAN}📊 OHLCV DATA SUMMARY${NC}"
echo "---------------------"

run_pretty_query "
SELECT 
    symbol,
    COUNT(*) as total_candles,
    MIN(timestamp) as oldest_candle,
    MAX(timestamp) as latest_candle,
    ROUND(EXTRACT(EPOCH FROM (MAX(timestamp) - MIN(timestamp))) / 3600) as hours_of_data,
    ROUND(EXTRACT(EPOCH FROM (NOW() - MAX(timestamp))) / 60) as minutes_since_last
FROM ohlcv_data
WHERE timeframe = '1h'
GROUP BY symbol
ORDER BY symbol;"

# Check for data gaps
echo -e "\n${CYAN}🔍 CHECKING FOR DATA GAPS${NC}"
echo "-------------------------"

for symbol in "BTC/USDT" "ETH/USDT" "SOL/USDT"; do
    GAPS=$(run_query "
    WITH hourly_series AS (
        SELECT generate_series(
            DATE_TRUNC('hour', MIN(timestamp)),
            DATE_TRUNC('hour', MAX(timestamp)),
            '1 hour'::interval
        ) AS expected_hour
        FROM ohlcv_data
        WHERE symbol = '$symbol' AND timeframe = '1h'
    ),
    actual_hours AS (
        SELECT DISTINCT DATE_TRUNC('hour', timestamp) as actual_hour
        FROM ohlcv_data
        WHERE symbol = '$symbol' AND timeframe = '1h'
    )
    SELECT COUNT(*)
    FROM hourly_series hs
    LEFT JOIN actual_hours ah ON hs.expected_hour = ah.actual_hour
    WHERE ah.actual_hour IS NULL;")
    
    if [ "$GAPS" = "0" ]; then
        echo -e "${GREEN}✅ $symbol: No gaps detected${NC}"
    else
        echo -e "${RED}❌ $symbol: $GAPS missing hours${NC}"
    fi
done

# Check futures data
echo -e "\n${CYAN}📈 FUTURES DATA SUMMARY${NC}"
echo "-----------------------"

run_pretty_query "
SELECT 
    symbol,
    COUNT(*) as total_records,
    MIN(timestamp) as oldest_record,
    MAX(timestamp) as latest_record,
    ROUND(AVG(open_interest)::numeric, 2) as avg_open_interest,
    ROUND(AVG(funding_rate)::numeric * 10000, 4) as avg_funding_rate_bps
FROM futures_data
GROUP BY symbol
ORDER BY symbol;"

# Check predictions data
echo -e "\n${CYAN}🔮 PREDICTIONS SUMMARY${NC}"
echo "----------------------"

PRED_COUNT=$(run_query "SELECT COUNT(*) FROM predictions;")

if [ "$PRED_COUNT" = "0" ] || [ -z "$PRED_COUNT" ]; then
    echo -e "${YELLOW}No predictions recorded yet${NC}"
else
    run_pretty_query "
    SELECT 
        symbol,
        ROUND(target_pct::numeric * 100, 0) || '%' as target,
        COUNT(*) as total_predictions,
        MIN(timestamp) as first_prediction,
        MAX(timestamp) as last_prediction
    FROM predictions
    GROUP BY symbol, target_pct
    ORDER BY symbol, target_pct;"
    
    # Check outcomes
    echo -e "\n${CYAN}📈 PREDICTION OUTCOMES${NC}"
    echo "----------------------"
    
    run_pretty_query "
    SELECT 
        p.symbol,
        ROUND(p.target_pct::numeric * 100, 0) || '%' as target,
        COUNT(DISTINCT p.id) as total_predictions,
        COUNT(po.id) as tracked_outcomes,
        SUM(CASE WHEN po.actual_outcome = p.prediction_class THEN 1 ELSE 0 END) as correct,
        CASE 
            WHEN COUNT(po.id) > 0 
            THEN ROUND(100.0 * SUM(CASE WHEN po.actual_outcome = p.prediction_class THEN 1 ELSE 0 END) / COUNT(po.id), 1) || '%'
            ELSE 'N/A' 
        END as accuracy
    FROM predictions p
    LEFT JOIN prediction_outcomes po ON p.id = po.prediction_id
    GROUP BY p.symbol, p.target_pct
    ORDER BY p.symbol, p.target_pct;"
fi

# Check model training history
echo -e "\n${CYAN}🤖 MODEL TRAINING HISTORY${NC}"
echo "-------------------------"

TRAINING_COUNT=$(run_query "SELECT COUNT(*) FROM model_training_history;")

if [ "$TRAINING_COUNT" = "0" ] || [ -z "$TRAINING_COUNT" ]; then
    echo -e "${YELLOW}No model training history recorded yet${NC}"
else
    run_pretty_query "
    SELECT 
        symbol,
        ROUND(target_pct::numeric * 100, 0) || '%' as target,
        COUNT(*) as trainings,
        MIN(trained_at) as first_training,
        MAX(trained_at) as latest_training,
        ROUND(AVG(final_accuracy)::numeric * 100, 1) || '%' as avg_accuracy,
        ROUND(AVG(training_samples)::numeric, 0) as avg_samples
    FROM model_training_history
    GROUP BY symbol, target_pct
    ORDER BY symbol, target_pct;"
    
    echo -e "\n${CYAN}📊 LATEST MODEL PERFORMANCE${NC}"
    echo "--------------------------"
    
    run_pretty_query "
    WITH latest_models AS (
        SELECT DISTINCT ON (symbol, target_pct)
            symbol,
            target_pct,
            trained_at,
            training_samples,
            cv_accuracy,
            cv_std,
            final_accuracy,
            precision,
            recall,
            f1_score
        FROM model_training_history
        ORDER BY symbol, target_pct, trained_at DESC
    )
    SELECT 
        symbol,
        ROUND(target_pct::numeric * 100, 0) || '%' as target,
        TO_CHAR(trained_at, 'YYYY-MM-DD HH24:MI') as trained_at,
        training_samples as samples,
        ROUND(cv_accuracy::numeric * 100, 1) || '%' as cv_acc,
        ROUND(cv_std::numeric * 100, 2) || '%' as cv_std,
        ROUND(final_accuracy::numeric * 100, 1) || '%' as final_acc,
        ROUND(f1_score::numeric, 3) as f1_score
    FROM latest_models
    ORDER BY symbol, target_pct;"
fi

# Database size information
echo -e "\n${CYAN}💾 DATABASE SIZE INFO${NC}"
echo "---------------------"

run_pretty_query "
SELECT 
    schemaname,
    tablename as table_name,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;"

# Total database size
TOTAL_SIZE=$(run_query "SELECT pg_size_pretty(pg_database_size('$DB_NAME'));")
echo -e "\n${GREEN}Total Database Size: $TOTAL_SIZE${NC}"

# Check table health
echo -e "\n${CYAN}🏥 TABLE HEALTH CHECK${NC}"
echo "---------------------"

# Check if all required tables exist
TABLES=("ohlcv_data" "futures_data" "feature_cache" "predictions" "prediction_outcomes" "model_performance" "telegram_reports" "system_health" "model_training_history")

for table in "${TABLES[@]}"; do
    EXISTS=$(run_query "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = '$table');")
    if [ "$EXISTS" = "t" ]; then
        COUNT=$(run_query "SELECT COUNT(*) FROM $table;")
        echo -e "${GREEN}✅ $table: exists (${COUNT} rows)${NC}"
    else
        echo -e "${RED}❌ $table: missing${NC}"
    fi
done

# Performance tips
echo -e "\n${YELLOW}💡 PERFORMANCE TIPS${NC}"
echo "-------------------"

# Check if hypertables are being used effectively
CHUNKS=$(run_query "SELECT COUNT(*) FROM timescaledb_information.chunks WHERE hypertable_name IN ('ohlcv_data', 'feature_cache');")
if [ -n "$CHUNKS" ] && [ "$CHUNKS" -gt "0" ]; then
    echo -e "${GREEN}✅ TimescaleDB chunks active: $CHUNKS chunks${NC}"
else
    echo -e "${YELLOW}⚠️  TimescaleDB might not be properly configured${NC}"
fi

# Check for missing indexes
echo -e "\n${CYAN}🔍 INDEX USAGE${NC}"
echo "--------------"

run_pretty_query "
SELECT 
    schemaname,
    relname as table_name,
    indexrelname as index_name,
    pg_size_pretty(pg_relation_size(indexrelid)) as index_size,
    idx_scan as times_used
FROM pg_stat_user_indexes
WHERE schemaname = 'public'
ORDER BY pg_relation_size(indexrelid) DESC
LIMIT 10;"