# BSE Predict - Hetzner VPS Deployment Guide (Git-Based)

This guide provides step-by-step instructions for deploying BSE Predict to a Hetzner VPS using a simple git-based approach.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Hetzner VPS Setup](#hetzner-vps-setup)
3. [Server Preparation](#server-preparation)
4. [Telegram Bot Setup](#telegram-bot-setup)
5. [Project Deployment](#project-deployment)
6. [Running the Application](#running-the-application)
7. [Monitoring and Maintenance](#monitoring-and-maintenance)
8. [Troubleshooting](#troubleshooting)
9. [Cost Analysis](#cost-analysis)

## Prerequisites

### Local Requirements
- SSH client (built-in on Linux/Mac, PuTTY on Windows)
- Your Telegram bot token and chat ID ready

### VPS Requirements
- Ubuntu 22.04 LTS (recommended)
- Minimum 2GB RAM (4GB recommended)
- 20GB+ disk space
- Root or sudo access

## Hetzner VPS Setup

### 1. Create a Hetzner Account
1. Visit [Hetzner Cloud](https://www.hetzner.com/cloud)
2. Create an account and add payment method
3. Navigate to the Cloud Console

### 2. Create a New Server

1. Click "New Server" in the Hetzner Cloud Console
2. Select configuration:
   - **Location**: Choose nearest to you (e.g., Nuremberg, Helsinki)
   - **Image**: Ubuntu 22.04
   - **Type**: CX21 (2 vCPU, 4GB RAM) - recommended for production
   - **Volume**: Optional, but recommended for database backups
   - **Network**: Default is fine
   - **SSH Keys**: Add your public SSH key (recommended)
   - **Name**: `bse-predict-prod` or similar

3. Click "Create & Buy now"

### 3. Note Your Server Details
- **IP Address**: You'll see this in the server list
- **Root Password**: Sent via email if you didn't use SSH keys

## Server Preparation

### 1. Connect to Your Server

```bash
ssh root@YOUR_SERVER_IP
```

### 2. Install Docker and Required Tools

```bash
# Update system
apt update && apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com | sh

# Install Docker Compose
apt install docker-compose -y

# Install other utilities
apt install git make htop -y
```

### 3. Setup Firewall (Optional but Recommended)

```bash
ufw allow ssh
ufw allow 8000/tcp  # Health check endpoint
ufw --force enable
```

## Telegram Bot Setup

### 1. Create a Telegram Bot

1. Open Telegram and search for [@BotFather](https://t.me/botfather)
2. Send `/newbot` command
3. Choose a name for your bot (e.g., "BSE Predict Alerts")
4. Choose a username (must end in 'bot', e.g., `bse_predict_alerts_bot`)
5. Save the bot token provided by BotFather

### 2. Get Your Chat ID

1. Start a chat with your new bot
2. Send any message to the bot
3. Visit: `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`
4. Look for `"chat":{"id":XXXXXXXXX}` - this is your chat ID
5. Save this chat ID

## Project Deployment

### 1. Clone the Repository

```bash
cd /opt
git clone https://github.com/YOUR_USERNAME/bse-predict.git
cd bse-predict
```

### 2. Create Environment File

```bash
# Copy the production template
cp .env.production .env

# Edit the file with your credentials
nano .env
```

Update the `.env` file with your values:
```env
# Database Configuration
POSTGRES_PASSWORD=your_secure_password_here

# Telegram Configuration
TELEGRAM_BOT_TOKEN=your_bot_token_from_botfather
TELEGRAM_CHAT_ID=your_chat_id

# Application Settings
LOG_LEVEL=INFO
ENVIRONMENT=production
```

### 3. Create Required Directories

```bash
mkdir -p logs models
```

## Running the Application

### Start the Application

Simply use the make command:

```bash
make run
```

This will:
1. Pull required Docker images
2. Build the application container
3. Start PostgreSQL/TimescaleDB
4. Start the BSE Predict application
5. Initialize the database
6. Begin data collection

### First Run

The first run will take longer (10-15 minutes) as it:
- Downloads historical data (up to 180 days)
- Trains initial ML models
- Sets up the database schema

Monitor the progress:
```bash
docker-compose -f docker-compose.prod.yml logs -f
```

### Verify Health

```bash
# Check container status
docker-compose -f docker-compose.prod.yml ps

# Check application health
curl http://localhost:8000/health
```

## Monitoring and Maintenance

### Daily Operations

The system runs automatically with:
- **Hourly**: Data fetching and predictions
- **Daily**: Model retraining (automated)

### Useful Commands

```bash
# View logs
docker-compose -f docker-compose.prod.yml logs -f

# Stop the application
docker-compose -f docker-compose.prod.yml down

# Restart the application
docker-compose -f docker-compose.prod.yml restart

# Update from git and restart
git pull
docker-compose -f docker-compose.prod.yml down
make run

# Force retrain models
make run RETRAIN=true
```

### Manual Backup

```bash
# Create backup directory
mkdir -p /opt/backups

# Backup database
docker exec bse-postgres-prod pg_dump -U crypto_user crypto_ml | gzip > /opt/backups/db_$(date +%Y%m%d).sql.gz

# Backup models
tar -czf /opt/backups/models_$(date +%Y%m%d).tar.gz -C /opt/bse-predict models/

# Backup environment
cp /opt/bse-predict/.env /opt/backups/env_$(date +%Y%m%d)
```

### Setup Automated Backups (Optional)

Create a backup script:
```bash
cat > /usr/local/bin/backup-bse.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/opt/backups"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# Backup database
docker exec bse-postgres-prod pg_dump -U crypto_user crypto_ml | gzip > $BACKUP_DIR/db_$DATE.sql.gz

# Backup models
tar -czf $BACKUP_DIR/models_$DATE.tar.gz -C /opt/bse-predict models/

# Keep only last 7 days
find $BACKUP_DIR -name "*" -mtime +7 -delete

echo "Backup completed: $DATE"
EOF

chmod +x /usr/local/bin/backup-bse.sh

# Add to crontab (daily at 2 AM)
(crontab -l 2>/dev/null; echo "0 2 * * * /usr/local/bin/backup-bse.sh > /var/log/bse-backup.log 2>&1") | crontab -
```

## Troubleshooting

### Common Issues

#### 1. Containers Won't Start
```bash
# Check logs
docker-compose -f docker-compose.prod.yml logs

# Check disk space
df -h

# Check memory
free -h
```

#### 2. Database Connection Issues
```bash
# Test database connection
docker exec bse-postgres-prod pg_isready -U crypto_user

# Check database logs
docker logs bse-postgres-prod
```

#### 3. Telegram Not Working
```bash
# Check environment variables
docker exec bse-predictor-prod env | grep TELEGRAM

# Test Telegram manually
docker exec bse-predictor-prod python -c "
from src.notifications.telegram_reporter import TelegramReporter
from src.utils.config import config
reporter = TelegramReporter(config.telegram_bot_token, config.telegram_chat_id)
reporter.send_test_message()
"
```

### Updating the Application

```bash
cd /opt/bse-predict

# Stop the application
docker-compose -f docker-compose.prod.yml down

# Pull latest changes
git pull

# Restart
make run
```

### Emergency Recovery

If something goes wrong:

```bash
# Stop everything
cd /opt/bse-predict
docker-compose -f docker-compose.prod.yml down

# Restore from backup (if available)
cd /opt/backups
gunzip -c db_YYYYMMDD.sql.gz | docker exec -i bse-postgres-prod psql -U crypto_user crypto_ml
tar -xzf models_YYYYMMDD.tar.gz -C /opt/bse-predict/

# Restart
cd /opt/bse-predict
make run
```

## Cost Analysis

### Hetzner VPS Costs (Monthly)

- **CX21** (2 vCPU, 4GB RAM, 40GB SSD): ~€5.83/month (recommended)
- **CX31** (2 vCPU, 8GB RAM, 80GB SSD): ~€10.59/month (if you need more resources)
- **Volume** (10GB backup storage): ~€0.52/month (optional)

**Total: €6-12/month** depending on configuration

### Resource Usage

Typical resource consumption:
- **CPU**: 10-30% average, spikes during training
- **RAM**: 2-3GB steady state
- **Disk**: ~5GB for application and data
- **Network**: Minimal (API calls only)

## Quick Reference

### Essential Commands
```bash
# Start application
cd /opt/bse-predict && make run

# View logs
docker-compose -f docker-compose.prod.yml logs -f

# Stop application
docker-compose -f docker-compose.prod.yml down

# Restart application
docker-compose -f docker-compose.prod.yml restart

# Check health
curl http://localhost:8000/health

# Update and restart
git pull && docker-compose -f docker-compose.prod.yml down && make run
```

### File Locations
- Project: `/opt/bse-predict/`
- Logs: `/opt/bse-predict/logs/`
- Models: `/opt/bse-predict/models/`
- Config: `/opt/bse-predict/.env`

### Container Names
- `bse-postgres-prod` - PostgreSQL database
- `bse-predictor-prod` - Main application

---

**That's it!** Your BSE Predict system should now be running on your Hetzner VPS. You should start receiving Telegram notifications within the next hour.