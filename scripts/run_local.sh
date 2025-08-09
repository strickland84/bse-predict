#!/bin/bash
# Run BSE Predict Full System Locally

set -e

echo "🚀 BSE Predict - Local Development"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Check if .env exists
if [ ! -f .env ]; then
    print_error ".env file not found!"
    echo "Please create .env file with your configuration"
    exit 1
fi

# Check if config.yaml exists
if [ ! -f config.yaml ]; then
    print_error "config.yaml file not found!"
    exit 1
fi

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down..."
    docker-compose -f docker-compose.dev.yml --env-file .env down
    print_status "Cleanup complete"
}

# Set trap to cleanup on exit
trap cleanup EXIT INT TERM

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Build and start the system
echo ""
echo "🔨 Building containers..."
docker-compose -f docker-compose.dev.yml --env-file .env build

echo ""
echo "🚀 Starting services..."
docker-compose -f docker-compose.dev.yml --env-file .env up -d

echo ""
echo "⏳ Waiting for services to initialize (30 seconds)..."
sleep 30

# Check health
echo ""
echo "💓 Checking system health..."
HEALTH_RESPONSE=$(curl -s http://localhost:8000/health 2>/dev/null || echo '{"status":"error"}')
HEALTH_STATUS=$(echo "$HEALTH_RESPONSE" | jq -r '.status' 2>/dev/null || echo 'unknown')

if [ "$HEALTH_STATUS" = "healthy" ]; then
    print_status "System is healthy!"
    echo "$HEALTH_RESPONSE" | jq . 2>/dev/null || echo "$HEALTH_RESPONSE"
else
    print_warning "System health check failed"
    echo "Response: $HEALTH_RESPONSE"
fi

# Show container status
echo ""
echo "📊 Container Status:"
docker-compose -f docker-compose.dev.yml --env-file .env ps

echo ""
echo "📈 Resource Usage:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# Show logs
echo ""
echo "📋 Recent Logs:"
docker-compose -f docker-compose.dev.yml --env-file .env logs --tail=20

echo ""
print_status "System is running!"
echo ""
echo "Useful commands:"
echo "- View logs: docker-compose -f docker-compose.dev.yml --env-file .env logs -f"
echo "- View specific service: docker-compose -f docker-compose.dev.yml --env-file .env logs -f bse-predictor"
echo "- Check health: curl http://localhost:8000/health | jq"
echo "- Stop system: docker-compose -f docker-compose.dev.yml --env-file .env down"
echo "- Restart service: docker-compose -f docker-compose.dev.yml --env-file .env restart bse-predictor"
echo ""
echo "Press Ctrl+C to stop the system"
echo ""

# Follow logs
docker-compose -f docker-compose.dev.yml --env-file .env logs -f