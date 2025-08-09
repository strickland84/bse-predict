# Multi-stage build for smaller production image
FROM python:3.11-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim

# Install runtime dependencies (include OpenMP and stdc++ for LightGBM/XGBoost wheels)
RUN apt-get update && apt-get install -y \
    curl \
    libgomp1 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PATH=/root/.local/bin:$PATH

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Copy all application files
COPY . .

# Create directories that the app needs
RUN mkdir -p /app/logs /app/models /app/data && \
    chmod -R 777 /app/logs /app/models /app/data

# Ensure Python can find our modules
RUN python -c "import sys; print('Python path:', sys.path)" && \
    ls -la /app/src/ && \
    ls -la /app/src/models/

# Expose health check port
EXPOSE 8000

# Run application directly with explicit PYTHONPATH
CMD ["python", "-u", "-m", "src.main"]
