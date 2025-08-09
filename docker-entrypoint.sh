#!/bin/bash
set -e

echo "🚀 Starting BSE Predict - Multi-Target Cryptocurrency Predictor..."

# Fix permissions for mounted volumes
echo "🔧 Fixing permissions for mounted volumes..."
if [ -d "/app/logs" ]; then
    # If logs directory exists (mounted), ensure we can write to it
    touch /app/logs/.test 2>/dev/null && rm /app/logs/.test 2>/dev/null || {
        echo "⚠️  Warning: /app/logs is not writable, using console logging only"
    }
fi

if [ -d "/app/models" ]; then
    # If models directory exists (mounted), ensure we can write to it
    touch /app/models/.test 2>/dev/null && rm /app/models/.test 2>/dev/null || {
        echo "❌ Error: /app/models is not writable, this is required for model storage"
        echo "Please ensure the models directory has proper permissions on the host"
        exit 1
    }
fi

# Start the application
exec python -u /app/src/main.py