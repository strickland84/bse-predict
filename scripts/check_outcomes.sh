#!/bin/bash
# Script to check and update prediction outcomes

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
    APP_CONTAINER="bse-predictor-prod"
else
    APP_CONTAINER="bse-predict-app-dev"
fi

# Default hours to check
HOURS=${1:-72}

# Check if container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${APP_CONTAINER}$"; then
    echo -e "${RED}Error: Container ${APP_CONTAINER} is not running!${NC}"
    echo -e "${YELLOW}Please start the ${ENVIRONMENT} environment first:${NC}"
    echo -e "${GREEN}make run${NC}"
    exit 1
fi

# Run the Python script inside the container
echo -e "${BLUE}🔍 Checking prediction outcomes (last ${HOURS} hours)...${NC}"
echo "============================================================"

# Execute the Python code directly
docker exec $APP_CONTAINER python -c "
from src.tracking.outcome_monitor import OutcomeMonitor
from src.database.prediction_tracker import prediction_tracker

# Initialize outcome monitor
monitor = OutcomeMonitor(monitoring_window_hours=${HOURS})

# Check outcomes
stats = monitor.check_prediction_outcomes()

# Display results
print(f'\\n📊 Outcome Check Results:')
print(f'  Total Checked: {stats[\"checked\"]} predictions')
print(f'  Completed: {stats[\"completed\"]}')
print(f'  ├─ Correct: {stats[\"hit_target\"]}')
print(f'  ├─ Incorrect: {stats[\"missed_target\"]}')
print(f'  └─ Expired: {stats[\"expired\"]}')
print(f'  Still Pending: {stats[\"still_pending\"]}')

# Show detailed results if available
if stats.get('completed_details'):
    print(f'\\n📋 Completed Predictions Detail:')
    print('-' * 60)
    
    # Group by result
    correct = [d for d in stats['completed_details'] if d['result'] == 'CORRECT']
    incorrect = [d for d in stats['completed_details'] if d['result'] == 'INCORRECT']
    expired = [d for d in stats['completed_details'] if d['result'] == 'EXPIRED']
    
    if correct:
        print(f'\\n✅ CORRECT ({len(correct)}):')
        for detail in correct[:10]:  # Show first 10
            time_str = f\" in {detail['time_to_target']:.1f}h\" if detail['time_to_target'] else \"\"
            print(f\"  • {detail['symbol']} {detail['target_pct']:.1%}: Predicted {detail['predicted']}{time_str}\")
    
    if incorrect:
        print(f'\\n❌ INCORRECT ({len(incorrect)}):')
        for detail in incorrect[:10]:
            time_str = f\" in {detail['time_to_target']:.1f}h\" if detail['time_to_target'] else \"\"
            print(f\"  • {detail['symbol']} {detail['target_pct']:.1%}: Predicted {detail['predicted']}{time_str}\")
    
    if expired:
        print(f'\\n⏰ EXPIRED ({len(expired)}):')
        for detail in expired[:10]:
            print(f\"  • {detail['symbol']} {detail['target_pct']:.1%}: Predicted {detail['predicted']}\")

# Get performance summary
print(f'\\n📈 Performance Summary (Last 24 hours):')
print('-' * 60)

perf = prediction_tracker.get_prediction_performance(days_back=1)
if not perf.empty:
    # Group by symbol
    symbols = perf['symbol'].unique()
    for symbol in sorted(symbols):
        symbol_perf = perf[perf['symbol'] == symbol]
        print(f'\\n{symbol}:')
        for _, row in symbol_perf.iterrows():
            if row['tracked_predictions'] > 0:
                accuracy = row['accuracy'] * 100 if row['accuracy'] else 0
                print(f\"  {row['target_pct']:.1%} target: {accuracy:.1f}% accuracy \"
                      f\"({int(row['correct_predictions'])}/{int(row['tracked_predictions'])} correct)\")
else:
    print('  No completed predictions in the last 24 hours')

# Get overall statistics
print(f'\\n📊 Overall Statistics (Last 7 days):')
print('-' * 60)

weekly_perf = prediction_tracker.get_prediction_performance(days_back=7)
if not weekly_perf.empty:
    total_predictions = weekly_perf['total_predictions'].sum()
    total_tracked = weekly_perf['tracked_predictions'].sum()
    total_correct = weekly_perf['correct_predictions'].sum()
    total_expired = weekly_perf['expired_predictions'].sum()
    
    if total_tracked > 0:
        overall_accuracy = (total_correct / (total_tracked - total_expired)) * 100 if (total_tracked - total_expired) > 0 else 0
        print(f'  Total Predictions Made: {int(total_predictions)}')
        print(f'  Total Outcomes Tracked: {int(total_tracked)}')
        print(f'  Overall Accuracy: {overall_accuracy:.1f}% ({int(total_correct)}/{int(total_tracked - total_expired)})')
        print(f'  Expired (no target hit): {int(total_expired)}')
        
        # Show by target percentage
        print(f'\\n  By Target Percentage:')
        target_summary = weekly_perf.groupby('target_pct').agg({
            'tracked_predictions': 'sum',
            'correct_predictions': 'sum',
            'expired_predictions': 'sum'
        })
        
        for target_pct, row in target_summary.iterrows():
            tracked = row['tracked_predictions']
            correct = row['correct_predictions']
            expired = row['expired_predictions']
            if tracked > 0:
                accuracy = (correct / (tracked - expired)) * 100 if (tracked - expired) > 0 else 0
                print(f'    {target_pct:.1%}: {accuracy:.1f}% accuracy ({int(correct)}/{int(tracked - expired)})')
else:
    print('  No predictions tracked in the last 7 days')
" 2>&1 | grep -v "INFO - "  # Filter out INFO log messages

echo ""
echo "============================================================"
echo -e "${GREEN}✅ Outcome check complete!${NC}"