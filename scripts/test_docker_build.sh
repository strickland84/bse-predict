#!/bin/bash
# Test script for Docker build and configuration

echo "🐳 Testing BSE Predict Docker Build"
echo "==================================="

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

# Create test environment file
echo "📝 Creating test environment file..."
cat > .env.test << 'EOF'
POSTGRES_PASSWORD=test_password_123
TELEGRAM_BOT_TOKEN=test_token
TELEGRAM_CHAT_ID=test_chat_id
LOG_LEVEL=DEBUG
ENVIRONMENT=test
EOF

# Test 1: Docker build
echo ""
echo "🔨 Building Docker image..."
if docker build -t bse-predictor:test . > /dev/null 2>&1; then
    print_status "Docker build successful"
else
    print_error "Docker build failed"
    exit 1
fi

# Test 2: Docker Compose configuration
echo ""
echo "🔧 Validating Docker Compose configuration..."
if docker-compose -f docker-compose.prod.yml --env-file .env.test config > /dev/null 2>&1; then
    print_status "Docker Compose configuration valid"
else
    print_error "Docker Compose configuration invalid"
    exit 1
fi

# Test 3: Start containers
echo ""
echo "🚀 Starting containers for testing..."
docker-compose -f docker-compose.prod.yml --env-file .env.test up -d

# Wait for containers to start
echo ""
echo "⏳ Waiting for services to initialize (30 seconds)..."
sleep 30

# Test 4: Check container status
echo ""
echo "📊 Checking container status..."
CONTAINERS_RUNNING=$(docker-compose -f docker-compose.prod.yml --env-file .env.test ps -q | wc -l)
if [ "$CONTAINERS_RUNNING" -eq "2" ]; then
    print_status "All containers running"
else
    print_error "Expected 2 containers, found $CONTAINERS_RUNNING"
fi

# Test 5: Database connection
echo ""
echo "🔌 Testing database connection..."
if docker exec bse-postgres-prod pg_isready -U crypto_user -d crypto_ml > /dev/null 2>&1; then
    print_status "Database is ready"
else
    print_error "Database not responding"
fi

# Test 6: Health check endpoint
echo ""
echo "💓 Testing health check endpoint..."
HEALTH_RESPONSE=$(curl -s http://localhost:8000/health 2>/dev/null || echo '{"status":"error"}')
if echo "$HEALTH_RESPONSE" | grep -q '"status"'; then
    print_status "Health endpoint responding"
    echo "Health status: $(echo $HEALTH_RESPONSE | jq -r '.status' 2>/dev/null || echo 'unknown')"
else
    print_warning "Health endpoint not ready yet"
fi

# Test 7: Check logs for errors
echo ""
echo "📋 Checking logs for errors..."
ERROR_COUNT=$(docker-compose -f docker-compose.prod.yml --env-file .env.test logs | grep -i error | wc -l)
if [ "$ERROR_COUNT" -gt "0" ]; then
    print_warning "Found $ERROR_COUNT error messages in logs"
    echo "Recent errors:"
    docker-compose -f docker-compose.prod.yml --env-file .env.test logs | grep -i error | tail -5
else
    print_status "No errors found in logs"
fi

# Show container resource usage
echo ""
echo "📈 Container resource usage:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# Cleanup
echo ""
echo "🧹 Cleaning up test environment..."
docker-compose -f docker-compose.prod.yml --env-file .env.test down -v
docker rmi bse-predictor:test > /dev/null 2>&1
rm -f .env.test

echo ""
echo "✅ Docker build tests completed!"
echo ""
echo "Next steps:"
echo "1. Copy .env.production to .env"
echo "2. Update .env with your Telegram credentials"
echo "3. Run: docker-compose -f docker-compose.prod.yml up -d"
echo "4. Monitor logs: docker-compose -f docker-compose.prod.yml logs -f"