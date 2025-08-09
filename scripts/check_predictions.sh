#!/bin/bash
# Script to check prediction performance from the command line

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
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

# Check if predictions table exists
check_tables() {
    EXISTS=$(run_query "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'predictions');")
    if [ "$EXISTS" != "t" ]; then
        echo -e "${RED}Error: Predictions table does not exist!${NC}"
        echo -e "${YELLOW}The prediction tracking tables need to be created.${NC}"
        echo -e "${YELLOW}Run the following command to create them:${NC}"
        echo -e "${GREEN}docker exec $DB_CONTAINER psql -U crypto_user -d crypto_ml -f /docker-entrypoint-initdb.d/init.sql${NC}"
        exit 1
    fi
}

case "$1" in
    "recent")
        # check_tables  # Skip for now since we know tables exist
        echo -e "${BLUE}=== Recent Predictions (Last 24h) ===${NC}"
        run_pretty_query "
        SELECT 
            p.symbol,
            p.target_pct::numeric(4,3) as target,
            TO_CHAR(p.timestamp, 'MM-DD HH24:MI') as time,
            CASE WHEN p.prediction_class = 1 THEN 'UP' ELSE 'DOWN' END as pred,
            ROUND(p.confidence::numeric * 100, 1) || '%' as conf,
            CASE 
                WHEN po.actual_outcome IS NULL THEN 'Pending'
                WHEN po.actual_outcome = p.prediction_class THEN 'CORRECT'
                ELSE 'WRONG'
            END as result
        FROM predictions p
        LEFT JOIN prediction_outcomes po ON p.id = po.prediction_id
        WHERE p.timestamp >= NOW() - INTERVAL '24 hours'
        ORDER BY p.timestamp DESC
        LIMIT 20;"
        ;;
        
    "accuracy")
        echo -e "${GREEN}=== Accuracy Summary (Last 7 Days) ===${NC}"
        run_pretty_query "
        SELECT 
            p.symbol,
            ROUND(p.target_pct::numeric * 100, 0) || '%' as target,
            COUNT(*) as total,
            COUNT(po.id) as done,
            SUM(CASE WHEN po.actual_outcome = p.prediction_class THEN 1 ELSE 0 END) as correct,
            CASE 
                WHEN COUNT(po.id) > 0 
                THEN ROUND(100.0 * SUM(CASE WHEN po.actual_outcome = p.prediction_class THEN 1 ELSE 0 END) / COUNT(po.id), 1) || '%'
                ELSE 'N/A' 
            END as accuracy
        FROM predictions p
        LEFT JOIN prediction_outcomes po ON p.id = po.prediction_id
        WHERE p.timestamp >= NOW() - INTERVAL '7 days'
        GROUP BY p.symbol, p.target_pct
        ORDER BY p.symbol, p.target_pct;"
        ;;
        
    "pending")
        echo -e "${YELLOW}=== Pending Predictions (Being Monitored) ===${NC}"
        run_pretty_query "
        SELECT 
            p.symbol,
            ROUND(p.target_pct::numeric * 100, 0) || '%' as target,
            CASE WHEN p.prediction_class = 1 THEN 'UP' ELSE 'DOWN' END as pred,
            ROUND(p.confidence::numeric * 100, 1) || '%' as conf,
            ROUND(EXTRACT(HOUR FROM NOW() - p.timestamp)) || 'h ago' as age
        FROM predictions p
        LEFT JOIN prediction_outcomes po ON p.id = po.prediction_id
        WHERE po.id IS NULL
            AND p.timestamp >= NOW() - INTERVAL '72 hours'
        ORDER BY p.timestamp DESC
        LIMIT 20;"
        ;;
        
    "highconf")
        echo -e "${GREEN}=== High Confidence Winners (>75% conf) ===${NC}"
        run_pretty_query "
        SELECT 
            p.symbol,
            ROUND(p.target_pct::numeric * 100, 0) || '%' as target,
            TO_CHAR(p.timestamp, 'MM-DD HH24:MI') as time,
            CASE WHEN p.prediction_class = 1 THEN 'UP' ELSE 'DOWN' END as pred,
            ROUND(p.confidence::numeric * 100, 1) || '%' as conf,
            ROUND(po.time_to_target_hours, 1) || 'h' as time_to_hit
        FROM predictions p
        JOIN prediction_outcomes po ON p.id = po.prediction_id
        WHERE p.confidence >= 0.75
            AND po.actual_outcome = p.prediction_class
            AND p.timestamp >= NOW() - INTERVAL '7 days'
        ORDER BY p.timestamp DESC
        LIMIT 20;"
        ;;
        
    "today")
        echo -e "${BLUE}=== Today's Performance ===${NC}"
        run_pretty_query "
        SELECT 
            p.symbol,
            COUNT(*) as predictions,
            COUNT(po.id) as completed,
            SUM(CASE WHEN po.actual_outcome = p.prediction_class THEN 1 ELSE 0 END) as correct,
            CASE 
                WHEN COUNT(po.id) > 0 
                THEN ROUND(100.0 * SUM(CASE WHEN po.actual_outcome = p.prediction_class THEN 1 ELSE 0 END) / COUNT(po.id), 1) || '%'
                ELSE 'N/A' 
            END as accuracy,
            ROUND(AVG(p.confidence)::numeric * 100, 1) || '%' as avg_conf
        FROM predictions p
        LEFT JOIN prediction_outcomes po ON p.id = po.prediction_id
        WHERE DATE(p.timestamp) = CURRENT_DATE
        GROUP BY p.symbol
        ORDER BY p.symbol;"
        ;;
        
    "stats")
        echo -e "${BLUE}=== Overall Statistics ===${NC}"
        
        # Total predictions
        TOTAL=$(run_query "SELECT COUNT(*) FROM predictions WHERE timestamp >= NOW() - INTERVAL '7 days';")
        echo -e "Total Predictions (7d): ${YELLOW}$TOTAL${NC}"
        
        # Completed predictions
        COMPLETED=$(run_query "SELECT COUNT(*) FROM predictions p JOIN prediction_outcomes po ON p.id = po.prediction_id WHERE p.timestamp >= NOW() - INTERVAL '7 days';")
        echo -e "Completed: ${GREEN}$COMPLETED${NC}"
        
        # Overall accuracy
        ACCURACY=$(run_query "
        SELECT ROUND(100.0 * SUM(CASE WHEN po.actual_outcome = p.prediction_class THEN 1 ELSE 0 END) / COUNT(po.id), 1)
        FROM predictions p
        JOIN prediction_outcomes po ON p.id = po.prediction_id
        WHERE p.timestamp >= NOW() - INTERVAL '7 days';")
        echo -e "Overall Accuracy: ${GREEN}${ACCURACY}%${NC}"
        
        # Best performer
        echo -e "\n${GREEN}Best Performer:${NC}"
        run_pretty_query "
        SELECT 
            p.symbol || ' ' || ROUND(p.target_pct::numeric * 100, 0) || '%' as asset_target,
            COUNT(po.id) as completed,
            ROUND(100.0 * SUM(CASE WHEN po.actual_outcome = p.prediction_class THEN 1 ELSE 0 END) / COUNT(po.id), 1) || '%' as accuracy
        FROM predictions p
        JOIN prediction_outcomes po ON p.id = po.prediction_id
        WHERE p.timestamp >= NOW() - INTERVAL '7 days'
        GROUP BY p.symbol, p.target_pct
        HAVING COUNT(po.id) >= 5
        ORDER BY (100.0 * SUM(CASE WHEN po.actual_outcome = p.prediction_class THEN 1 ELSE 0 END) / COUNT(po.id)) DESC
        LIMIT 3;"
        ;;
        
    *)
        echo "BSE Predict - Prediction Performance Checker"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  recent    - Show recent predictions from last 24 hours"
        echo "  accuracy  - Show accuracy summary by symbol and target"
        echo "  pending   - Show predictions still being monitored"
        echo "  highconf  - Show high confidence winners (>75%)"
        echo "  today     - Show today's performance summary"
        echo "  stats     - Show overall statistics"
        echo ""
        echo "Example: $0 recent"
        ;;
esac