# PHASE 4: CONTAINERIZATION & DEPLOYMENT (Days 9-12)
**🎯 GOAL**: Deploy production-ready system to Hetzner VPS

## What You're Building
- Docker containers for easy deployment and scaling
- Production-grade configuration management
- Automated deployment scripts for Hetzner VPS
- Health monitoring and backup systems
- SSL setup and firewall configuration

---

## CHECKPOINT 4A: Docker Setup (Day 9, 6 hours)

### Step 4.1: Production Dockerfile
```dockerfile
# Dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config.yaml .
COPY sql/ ./sql/

# Create necessary directories
RUN mkdir -p {models,logs,data}

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)"

# Create startup script
COPY <<'EOF' /app/start.sh
#!/bin/bash
set -e

echo "🚀 Starting Crypto ML Multi-Target Predictor..."

# Wait for database
echo "⏳ Waiting for database..."
python -c "
import time
import sys
from src.database.connection import DatabaseConnection
from src.utils.config import Config

config = Config()
db = DatabaseConnection(config.database_url)

for i in range(30):
    if db.test_connection():
        print('✅ Database connected')
        break
    print(f'⏳ Waiting for database... ({i+1}/30)')
    time.sleep(2)
else:
    print('❌ Database connection failed')
    sys.exit(1)
"

# Create tables
echo "🏗️ Creating database tables..."
python -c "
from src.database.connection import DatabaseConnection
from src.utils.config import Config

config = Config()
db = DatabaseConnection(config.database_url)
db.create_tables()
print('✅ Tables created')
"

# Start application
echo "🎯 Starting main application..."
python src/main.py
EOF

RUN chmod +x /app/start.sh

# Run application
CMD ["/app/start.sh"]
```

### Step 4.2: Production Docker Compose
```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  postgres:
    image: timescale/timescaledb:latest-pg15
    container_name: crypto-postgres-prod
    restart: unless-stopped
    environment:
      - POSTGRES_DB=crypto_ml
      - POSTGRES_USER=crypto_user
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      - POSTGRES_INITDB_ARGS=--auth-host=scram-sha-256
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./sql/init.sql:/docker-entrypoint-initdb.d/init.sql
      - ./postgresql.conf:/etc/postgresql/postgresql.conf
    ports:
      - "127.0.0.1:5432:5432"  # Only bind to localhost
    networks:
      - crypto-network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U crypto_user -d crypto_ml"]
      interval: 30s
      timeout: 10s
      retries: 3

  crypto-predictor:
    build: .
    container_name: crypto-ml-predictor-prod
    restart: unless-stopped
    depends_on:
      postgres:
        condition: service_healthy
    environment:
      - TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}
      - TELEGRAM_CHAT_ID=${TELEGRAM_CHAT_ID}
      - DATABASE_URL=postgresql://crypto_user:${POSTGRES_PASSWORD}@postgres:5432/crypto_ml
      - LOG_LEVEL=INFO
      - ENVIRONMENT=production
    volumes:
      - ./logs:/app/logs
      - model_data:/app/models
    ports:
      - "127.0.0.1:8000:8000"  # Health check endpoint
    networks:
      - crypto-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 60s
      timeout: 30s
      retries: 3
      start_period: 120s

volumes:
  postgres_data:
    driver: local
  model_data:
    driver: local

networks:
  crypto-network:
    driver: bridge
```

### Step 4.3: Main Application Entry Point
```python
# src/main.py
import logging
import signal
import sys
import os
from flask import Flask, jsonify
import threading
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class CryptoMLApplication:
    def __init__(self):
        self.scheduler = None
        self.flask_app = None
        self.is_running = False
        
    def initialize(self):
        """Initialize application components"""
        try:
            logger.info("🚀 Initializing Crypto ML Multi-Target Predictor")
            
            # Load configuration
            from src.utils.config import Config
            self.config = Config()
            logger.info("✅ Configuration loaded")
            
            # Initialize database
            from src.database.operations import DatabaseOperations
            self.db_ops = DatabaseOperations(self.config.database_url)
            logger.info("✅ Database connection established")
            
            # Initialize scheduler
            from src.scheduler.task_scheduler import MultiTargetTaskScheduler
            self.scheduler = MultiTargetTaskScheduler(self.db_ops, self.config)
            logger.info("✅ Task scheduler initialized")
            
            # Setup Flask health check endpoint
            self.setup_health_check()
            logger.info("✅ Health check endpoint configured")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            return False
            
    def setup_health_check(self):
        """Setup Flask health check endpoint"""
        self.flask_app = Flask(__name__)
        
        @self.flask_app.route('/health')
        def health_check():
            try:
                # Check database connection
                db_status = self.db_ops.db.test_connection()
                
                # Check scheduler status
                scheduler_status = self.scheduler.is_running if self.scheduler else False
                
                # Check if we have recent predictions
                recent_predictions = True  # This would check for predictions in last 2 hours
                
                status = {
                    'status': 'healthy' if (db_status and scheduler_status) else 'unhealthy',
                    'timestamp': datetime.now().isoformat(),
                    'components': {
                        'database': 'up' if db_status else 'down',
                        'scheduler': 'running' if scheduler_status else 'stopped',
                        'predictions': 'current' if recent_predictions else 'stale'
                    },
                    'uptime_seconds': int(time.time() - self.start_time) if hasattr(self, 'start_time') else 0
                }
                
                return jsonify(status), 200 if status['status'] == 'healthy' else 503
                
            except Exception as e:
                return jsonify({
                    'status': 'error',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
                
    def start_health_server(self):
        """Start Flask health check server"""
        self.flask_app.run(host='0.0.0.0', port=8000, debug=False, use_reloader=False)
        
    def run(self):
        """Run the application"""
        if not self.initialize():
            logger.error("❌ Failed to initialize application")
            sys.exit(1)
            
        try:
            self.start_time = time.time()
            self.is_running = True
            
            # Start health check server in separate thread
            health_thread = threading.Thread(target=self.start_health_server, daemon=True)
            health_thread.start()
            logger.info("✅ Health check server started on port 8000")
            
            # Initial data fetch and model training
            logger.info("🔄 Running initial setup...")
            self.run_initial_setup()
            
            # Start scheduler
            logger.info("⏰ Starting task scheduler...")
            self.scheduler.start()
            
            # Run initial prediction
            logger.info("🔮 Running initial predictions...")
            self.scheduler.run_manual_prediction()
            
            logger.info("🎉 Application started successfully!")
            logger.info("📱 Hourly reports will be sent to Telegram")
            logger.info("🔄 Models will retrain daily at 06:00 UTC")
            
            # Keep main thread alive
            while self.is_running:
                time.sleep(60)
                
        except KeyboardInterrupt:
            logger.info("🛑 Shutdown requested by user")
        except Exception as e:
            logger.error(f"❌ Application error: {e}")
        finally:
            self.shutdown()
            
    def run_initial_setup(self):
        """Run initial data fetch and model training with recovery"""
        try:
            # Initialize data recovery manager
            from src.data.recovery import DataRecoveryManager
            recovery_manager = DataRecoveryManager(self.db_ops, self.scheduler.data_fetcher)
            
            # Check and recover data integrity
            logger.info("🔍 Checking data integrity...")
            recovery_results = recovery_manager.check_and_recover_all_symbols(
                self.config.assets, 
                hours_back=168  # Check last week
            )
            
            for symbol, filled in recovery_results.items():
                if filled > 0:
                    logger.info(f"✅ Recovered {filled} missing candles for {symbol}")
                    
            # Ensure minimum data coverage
            logger.info("📊 Ensuring minimum data coverage...")
            for symbol in self.config.assets:
                if not recovery_manager.ensure_minimum_data_coverage(symbol, min_hours=168):
                    logger.warning(f"⚠️ Could not ensure minimum data for {symbol}")
                
            # Train initial models if not exist
            logger.info("🏋️ Training initial models...")
            self.scheduler.trainer.train_all_models(self.config.assets, retrain=False)
            
        except Exception as e:
            logger.error(f"❌ Initial setup failed: {e}")
            
    def shutdown(self):
        """Graceful shutdown"""
        logger.info("🛑 Shutting down application...")
        self.is_running = False
        
        if self.scheduler:
            self.scheduler.stop()
            
        logger.info("✅ Application shutdown complete")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"🛑 Received signal {signum}")
    sys.exit(0)

if __name__ == "__main__":
    # Setup signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create and run application
    app = CryptoMLApplication()
    app.run()
```

### Step 4.4: Environment Configuration
```bash
# .env.production
POSTGRES_PASSWORD=$(openssl rand -base64 32)
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
DATABASE_URL=postgresql://crypto_user:${POSTGRES_PASSWORD}@postgres:5432/crypto_ml
LOG_LEVEL=INFO
ENVIRONMENT=production
```

**🔍 CHECKPOINT 4A TEST:**
```bash
# test_docker_build.sh
#!/bin/bash

echo "🐳 Testing Docker build..."

# Create test environment file
cat > .env.test << 'EOF'
POSTGRES_PASSWORD=test_password_123
TELEGRAM_BOT_TOKEN=test_token
TELEGRAM_CHAT_ID=test_chat_id
EOF

# Build Docker image
docker build -t crypto-ml-predictor:test .

if [ $? -eq 0 ]; then
    echo "✅ Docker build successful"
    
    # Test docker-compose
    docker-compose -f docker-compose.prod.yml --env-file .env.test config
    
    if [ $? -eq 0 ]; then
        echo "✅ Docker Compose configuration valid"
    else
        echo "❌ Docker Compose configuration invalid"
    fi
else
    echo "❌ Docker build failed"
fi

# Cleanup
rm -f .env.test
```

---

## CHECKPOINT 4B: Hetzner Deployment (Day 10-11, 8 hours)

### Step 4.5: Hetzner VPS Setup Script
```bash
#!/bin/bash
# deploy_to_hetzner.sh

set -e

echo "🚀 Deploying Crypto ML Predictor to Hetzner VPS"
echo "=================================================="

# Configuration
VPS_IP="YOUR_VPS_IP"
VPS_USER="root"
PROJECT_DIR="/opt/crypto-ml-predictor"
REPO_URL="https://github.com/yourusername/crypto-ml-predictor.git"

# Function to run commands on VPS
run_on_vps() {
    ssh $VPS_USER@$VPS_IP "$1"
}

# Function to copy files to VPS
copy_to_vps() {
    scp -r "$1" $VPS_USER@$VPS_IP:"$2"
}

echo "📡 Testing VPS connection..."
if run_on_vps "echo 'Connection successful'"; then
    echo "✅ VPS connection established"
else
    echo "❌ Cannot connect to VPS"
    exit 1
fi

echo "🔧 Setting up VPS environment..."
run_on_vps "
    # Update system
    apt update && apt upgrade -y
    
    # Install Docker
    if ! command -v docker &> /dev/null; then
        curl -fsSL https://get.docker.com -o get-docker.sh
        sh get-docker.sh
        rm get-docker.sh
    fi
    
    # Install Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        curl -L \"https://github.com/docker/compose/releases/latest/download/docker-compose-\$(uname -s)-\$(uname -m)\" -o /usr/local/bin/docker-compose
        chmod +x /usr/local/bin/docker-compose
    fi
    
    # Install other utilities
    apt install -y git curl htop nano ufw
    
    # Setup firewall
    ufw --force reset
    ufw default deny incoming
    ufw default allow outgoing
    ufw allow ssh
    ufw allow 8000  # Health check
    ufw --force enable
    
    echo '✅ VPS environment setup complete'
"

echo "📁 Setting up project directory..."
run_on_vps "
    # Remove existing directory if it exists
    rm -rf $PROJECT_DIR
    
    # Create project directory
    mkdir -p $PROJECT_DIR
    cd $PROJECT_DIR
    
    echo '✅ Project directory created'
"

echo "📂 Copying project files..."
copy_to_vps "." "$PROJECT_DIR/"

echo "🔧 Setting up production configuration..."
run_on_vps "
    cd $PROJECT_DIR
    
    # Create production environment file
    cat > .env << 'EOF'
POSTGRES_PASSWORD=\$(openssl rand -base64 32)
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
EOF

    # Create logs directory
    mkdir -p logs
    
    # Set permissions
    chmod +x deploy_to_hetzner.sh
    
    echo '✅ Production configuration ready'
"

echo "🐳 Building and starting containers..."
run_on_vps "
    cd $PROJECT_DIR
    
    # Build images
    docker-compose -f docker-compose.prod.yml build
    
    # Start services
    docker-compose -f docker-compose.prod.yml up -d
    
    echo '✅ Containers started'
"

echo "⏳ Waiting for services to start..."
sleep 30

echo "🔍 Checking service health..."
run_on_vps "
    cd $PROJECT_DIR
    
    # Check container status
    docker-compose -f docker-compose.prod.yml ps
    
    # Check health
    curl -f http://localhost:8000/health || echo 'Health check not ready yet'
    
    # Show logs
    echo '📋 Recent logs:'
    docker-compose -f docker-compose.prod.yml logs --tail=20
"

echo "📋 Setting up monitoring and maintenance..."
run_on_vps "
    # Setup log rotation
    cat > /etc/logrotate.d/crypto-ml << 'EOF'
$PROJECT_DIR/logs/*.log {
    daily
    rotate 30
    compress
    missingok
    notifempty
    create 644 root root
    postrotate
        docker-compose -f $PROJECT_DIR/docker-compose.prod.yml restart crypto-predictor
    endscript
}
EOF

    # Setup daily backup script
    cat > /usr/local/bin/backup-crypto-ml.sh << 'EOF'
#!/bin/bash
BACKUP_DIR=\"/root/backups/crypto-ml\"
DATE=\$(date +%Y%m%d_%H%M%S)

mkdir -p \$BACKUP_DIR

# Backup database
docker exec crypto-postgres-prod pg_dump -U crypto_user crypto_ml > \$BACKUP_DIR/db_backup_\$DATE.sql

# Backup models
tar -czf \$BACKUP_DIR/models_backup_\$DATE.tar.gz -C $PROJECT_DIR models/

# Cleanup old backups (keep 7 days)
find \$BACKUP_DIR -name \"*backup*\" -mtime +7 -delete

echo \"Backup completed: \$DATE\"
EOF

    chmod +x /usr/local/bin/backup-crypto-ml.sh
    
    # Add to crontab
    (crontab -l 2>/dev/null; echo \"0 2 * * * /usr/local/bin/backup-crypto-ml.sh\") | crontab -
    
    echo '✅ Monitoring and backup setup complete'
"

echo "🎉 Deployment complete!"
echo "=================================================="
echo "🔗 Health check: http://$VPS_IP:8000/health"
echo "📱 Configure Telegram bot token in $PROJECT_DIR/.env"
echo "📊 View logs: docker-compose -f $PROJECT_DIR/docker-compose.prod.yml logs -f"
echo "🔄 Restart: docker-compose -f $PROJECT_DIR/docker-compose.prod.yml restart"
echo "📈 Monitor: htop, docker stats"

echo "⚠️  IMPORTANT NEXT STEPS:"
echo "1. Edit $PROJECT_DIR/.env with your Telegram credentials"
echo "2. Restart services: docker-compose -f $PROJECT_DIR/docker-compose.prod.yml restart"
echo "3. Monitor initial data fetch and model training in logs"
echo "4. Test Telegram integration"
echo "5. Setup external monitoring (optional)"
```

### Step 4.6: PostgreSQL Configuration Optimization
```bash
# postgresql.conf (for production optimization)
# Optimized for 8GB RAM VPS
shared_buffers = 2GB
effective_cache_size = 6GB
maintenance_work_mem = 512MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200
work_mem = 64MB
max_connections = 100

# TimescaleDB optimizations
timescaledb.max_background_workers = 8
shared_preload_libraries = 'timescaledb'

# Logging
log_statement = 'none'  # Reduce log noise
log_min_duration_statement = 1000  # Log slow queries only
```

**🔍 CHECKPOINT 4B TEST:**
```bash
# After deployment, test on VPS
ssh root@YOUR_VPS_IP "
    cd /opt/crypto-ml-predictor
    
    # Check service status
    docker-compose -f docker-compose.prod.yml ps
    
    # Test health endpoint
    curl -s http://localhost:8000/health | python3 -m json.tool
    
    # Check recent logs
    docker-compose -f docker-compose.prod.yml logs --tail=50 crypto-predictor
"
```

---

## Phase 4 Success Criteria

After completing Phase 4, you should have:

✅ **Dockerized application** running in containers  
✅ **Production deployment** on Hetzner VPS  
✅ **Health monitoring** endpoint responding  
✅ **Automated backups** scheduled daily  
✅ **Firewall configured** with proper security  
✅ **Log rotation** and maintenance scripts  

**Next Step**: Move to Phase 5 for final testing and optimization.
