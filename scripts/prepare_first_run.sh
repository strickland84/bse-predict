#!/bin/bash
# Prepare BSE Predict for first production run

set -e

echo "🔧 BSE Predict - First Run Preparation"
echo "====================================="

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

# Check prerequisites
echo "📋 Checking prerequisites..."

# Check Docker
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed!"
    echo "Please install Docker from https://docker.com"
    exit 1
fi
print_status "Docker installed"

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed!"
    exit 1
fi
print_status "Docker Compose installed"

# Check .env file
if [ ! -f .env ]; then
    print_warning ".env file not found - creating from template"
    if [ -f .env.production ]; then
        cp .env.production .env
        print_status ".env created from template"
        echo ""
        print_warning "IMPORTANT: Edit .env and add your credentials:"
        echo "  - POSTGRES_PASSWORD (or keep generated one)"
        echo "  - TELEGRAM_BOT_TOKEN"
        echo "  - TELEGRAM_CHAT_ID"
        echo ""
        echo "Press Enter to continue after updating .env..."
        read
    else
        print_error ".env.production template not found!"
        exit 1
    fi
else
    print_status ".env file exists"
fi

# Validate .env has required values
echo ""
echo "🔍 Validating configuration..."
source .env

if [ -z "$TELEGRAM_BOT_TOKEN" ] || [ "$TELEGRAM_BOT_TOKEN" = "YOUR_BOT_TOKEN_HERE" ]; then
    print_warning "TELEGRAM_BOT_TOKEN not configured"
    echo "The system will run but won't send Telegram notifications"
fi

if [ -z "$TELEGRAM_CHAT_ID" ] || [ "$TELEGRAM_CHAT_ID" = "YOUR_CHAT_ID_HERE" ]; then
    print_warning "TELEGRAM_CHAT_ID not configured"
    echo "The system will run but won't send Telegram notifications"
fi

# Create necessary directories
echo ""
echo "📁 Creating directories..."
mkdir -p logs models
print_status "Directories created"

# Pull base images
echo ""
echo "🐳 Pulling Docker images..."
docker pull timescale/timescaledb:latest-pg15
docker pull python:3.11-slim
print_status "Base images pulled"

# Test database connection with dev setup first
echo ""
echo "🔌 Testing database setup..."
if docker-compose -f docker-compose.dev.yml ps | grep -q "Up"; then
    print_warning "Development database is running"
    echo "Do you want to stop it? (y/n)"
    read -r response
    if [ "$response" = "y" ]; then
        docker-compose -f docker-compose.dev.yml down
        print_status "Development database stopped"
    fi
fi

echo ""
print_status "System is ready for production run!"
echo ""
echo "To start the system, run:"
echo "  ./scripts/run_local_prod.sh"
echo ""
echo "This will:"
echo "1. Build the production Docker image"
echo "2. Start PostgreSQL/TimescaleDB"
echo "3. Start the BSE Predict application"
echo "4. Initialize the database"
echo "5. Begin data collection and predictions"
echo ""
print_warning "First run will take longer as it:"
echo "- Downloads historical data (up to 180 days)"
echo "- Trains initial ML models"
echo "- Sets up the database schema"