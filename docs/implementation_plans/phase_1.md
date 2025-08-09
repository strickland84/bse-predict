# PHASE 1: FOUNDATION SETUP (Days 1-3)
**🎯 GOAL**: Establish core infrastructure and basic data pipeline

## What You're Building
- Project structure with proper configuration management
- PostgreSQL + TimescaleDB database for time-series data
- Exchange data fetcher using CCXT
- Basic database operations for storing/retrieving data
- Configuration system for easy environment management

---

## CHECKPOINT 1A: Project Structure & Environment (Day 1, 4 hours)

### Step 1.1: Create Project Structure
```bash
# Initialize project
mkdir crypto-ml-predictor
cd crypto-ml-predictor

# Create directory structure
mkdir -p {src/{data,database,models,utils,notifications,scheduler},sql,logs,tests,monitoring}
mkdir -p src/database/migrations

# Initialize git
git init
```

### Step 1.2: Setup Development Environment
```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Create requirements.txt
cat > requirements.txt << 'EOF'
# Core ML and Data
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
xgboost==1.7.6

# Database
psycopg2-binary==2.9.7
sqlalchemy==2.0.21
alembic==1.12.0

# Exchange API
ccxt==4.0.67

# Telegram
python-telegram-bot==20.5

# Utilities
schedule==1.2.0
python-dotenv==1.0.0
pyyaml==6.0.1
flask==2.3.3
psutil==5.9.5

# Development
pytest==7.4.2
black==23.7.0
flake8==6.0.0
EOF

pip install -r requirements.txt
```

### Step 1.3: Configuration Management
```python
# src/utils/config.py
import os
import yaml
from pathlib import Path

class Config:
    def __init__(self):
        self.config = self.load_config()
        
    def load_config(self):
        """Load configuration from YAML and environment variables"""
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Override with environment variables
        config['database']['url'] = os.getenv('DATABASE_URL', config['database']['url'])
        config['telegram']['bot_token'] = os.getenv('TELEGRAM_BOT_TOKEN', '')
        config['telegram']['chat_id'] = os.getenv('TELEGRAM_CHAT_ID', '')
        
        return config
    
    @property
    def assets(self):
        return self.config['trading']['assets']
    
    @property 
    def target_percentages(self):
        return self.config['trading']['target_percentages']
    
    @property
    def database_url(self):
        return self.config['database']['url']
```

```yaml
# config.yaml
app:
  name: "Crypto ML Multi-Target Predictor"
  version: "1.0.0"
  log_level: "INFO"

trading:
  assets:
    - "BTC/USDT"
    - "ETH/USDT" 
    - "SOL/USDT"
  target_percentages: [0.01, 0.02, 0.05]  # 1%, 2%, 5%
  timeframe: "1h"
  lookback_days: 180

database:
  url: "postgresql://crypto_user:password@localhost:5432/crypto_ml"
  
ml:
  retrain_frequency: "daily"
  min_samples_for_training: 500
  max_lookback_hours: 168  # 1 week
  
telegram:
  bot_token: ""
  chat_id: ""
  report_frequency: "hourly"
  alert_threshold: 0.75
```

**🔍 CHECKPOINT 1A TEST:**
```bash
# Test basic setup
python -c "from src.utils.config import Config; c = Config(); print('Config loaded:', c.assets)"
# Expected: Config loaded: ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
```

---

## CHECKPOINT 1B: Database Setup (Day 1, 3 hours)

### Step 1.4: Database Schema Design
```sql
-- sql/init.sql
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- OHLCV candle data (time-series optimized)
CREATE TABLE ohlcv_data (
    id BIGSERIAL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    open DECIMAL(20,8) NOT NULL,
    high DECIMAL(20,8) NOT NULL,
    low DECIMAL(20,8) NOT NULL,
    close DECIMAL(20,8) NOT NULL,
    volume DECIMAL(20,8) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (symbol, timeframe, timestamp)
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('ohlcv_data', 'timestamp', 'symbol', 3);

-- Feature data (computed features cached in database)
CREATE TABLE feature_cache (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    feature_set JSONB NOT NULL, -- Store all features as JSON for flexibility
    feature_version VARCHAR(10) NOT NULL, -- Track feature engineering versions
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(symbol, timestamp, feature_version)
);

-- Create hypertable for feature cache too
SELECT create_hypertable('feature_cache', 'timestamp', 'symbol', 3);

-- Predictions table (modified for multi-target)
CREATE TABLE predictions (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    target_pct DECIMAL(4,3) NOT NULL, -- 0.01, 0.02, 0.05
    timestamp TIMESTAMPTZ NOT NULL,
    model_name VARCHAR(50) NOT NULL,
    prediction_class INTEGER NOT NULL, -- 0 for DOWN, 1 for UP
    probability DECIMAL(5,4) NOT NULL,
    confidence DECIMAL(5,4) NOT NULL,
    features_used JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(symbol, target_pct, timestamp, model_name)
);

-- Prediction outcomes (modified for multi-target tracking)
CREATE TABLE prediction_outcomes (
    id BIGSERIAL PRIMARY KEY,
    prediction_id BIGINT REFERENCES predictions(id),
    actual_outcome INTEGER, -- 0 for DOWN, 1 for UP, NULL if incomplete
    target_hit_timestamp TIMESTAMPTZ,
    time_to_target_hours DECIMAL(6,2),
    max_favorable_move DECIMAL(6,4),
    max_adverse_move DECIMAL(6,4),
    completed_at TIMESTAMPTZ DEFAULT NOW()
);

-- Model performance metrics (modified for multi-target)
CREATE TABLE model_performance (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    target_pct DECIMAL(4,3) NOT NULL,
    model_name VARCHAR(50) NOT NULL,
    evaluation_date DATE NOT NULL,
    accuracy DECIMAL(5,4),
    precision_up DECIMAL(5,4),
    precision_down DECIMAL(5,4),
    recall_up DECIMAL(5,4),
    recall_down DECIMAL(5,4),
    f1_up DECIMAL(5,4),
    f1_down DECIMAL(5,4),
    total_predictions INTEGER,
    avg_time_to_target_hours DECIMAL(6,2),
    hit_rate DECIMAL(5,4), -- Percentage of predictions that hit either target
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(symbol, target_pct, model_name, evaluation_date)
);

-- Telegram reports log
CREATE TABLE telegram_reports (
    id BIGSERIAL PRIMARY KEY,
    report_type VARCHAR(20) NOT NULL, -- 'hourly', 'daily', 'alert'
    content TEXT NOT NULL,
    sent_at TIMESTAMPTZ DEFAULT NOW(),
    success BOOLEAN DEFAULT FALSE,
    error_message TEXT
);

-- System health metrics
CREATE TABLE system_health (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    cpu_percent DECIMAL(5,2),
    memory_percent DECIMAL(5,2),
    disk_percent DECIMAL(5,2),
    prediction_latency_ms INTEGER,
    data_freshness_minutes INTEGER,
    active_models INTEGER
);

-- Create indexes for performance
CREATE INDEX idx_ohlcv_symbol_time ON ohlcv_data(symbol, timestamp DESC);
CREATE INDEX idx_feature_cache_symbol_time ON feature_cache(symbol, timestamp DESC);
CREATE INDEX idx_predictions_symbol_target_time ON predictions(symbol, target_pct, timestamp DESC);
CREATE INDEX idx_outcomes_prediction ON prediction_outcomes(prediction_id);
CREATE INDEX idx_performance_symbol_target_model ON model_performance(symbol, target_pct, model_name, evaluation_date DESC);
CREATE INDEX idx_health_timestamp ON system_health(timestamp DESC);

-- Performance optimization: Create materialized view for recent data
CREATE MATERIALIZED VIEW recent_ohlcv AS
SELECT * FROM ohlcv_data 
WHERE timestamp >= NOW() - INTERVAL '7 days'
ORDER BY symbol, timestamp DESC;

CREATE UNIQUE INDEX idx_recent_ohlcv_unique ON recent_ohlcv(symbol, timeframe, timestamp);
```

### Step 1.5: Database Connection Module
```python
# src/database/connection.py
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import logging

Base = declarative_base()

class DatabaseConnection:
    def __init__(self, database_url):
        self.database_url = database_url
        self.engine = create_engine(
            database_url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            echo=False  # Set to True for SQL debugging
        )
        self.SessionLocal = sessionmaker(bind=self.engine)
        
    def create_tables(self):
        """Create all tables"""
        Base.metadata.create_all(bind=self.engine)
        
    def test_connection(self):
        """Test database connectivity"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logging.error(f"Database connection failed: {e}")
            return False
            
    def get_session(self):
        """Get database session"""
        return self.SessionLocal()
```

### Step 1.6: Docker Database Setup
```yaml
# docker-compose.dev.yml (for development)
version: '3.8'

services:
  postgres-dev:
    image: timescale/timescaledb:latest-pg15
    container_name: crypto-postgres-dev
    environment:
      - POSTGRES_DB=crypto_ml
      - POSTGRES_USER=crypto_user
      - POSTGRES_PASSWORD=dev_password
    volumes:
      - postgres_dev_data:/var/lib/postgresql/data
      - ./sql/init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"

volumes:
  postgres_dev_data:
```

```bash
# Start development database
docker-compose -f docker-compose.dev.yml up -d postgres-dev
```

**🔍 CHECKPOINT 1B TEST:**
```python
# test_database.py
from src.database.connection import DatabaseConnection
from src.utils.config import Config

config = Config()
db = DatabaseConnection(config.database_url)

# Test connection
assert db.test_connection(), "Database connection failed"

# Create tables
db.create_tables()
print("✅ Database setup complete")
```

---

## CHECKPOINT 1C: Basic Data Fetching (Day 2, 4 hours)

### Step 1.7: Exchange Data Fetcher
```python
# src/data/fetcher.py
import ccxt
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import List, Optional
import time
import logging

class ExchangeDataFetcher:
    def __init__(self, exchange_name='binance'):
        self.exchange = getattr(ccxt, exchange_name)({
            'apiKey': '',  # Not needed for public data
            'secret': '',
            'timeout': 30000,
            'enableRateLimit': True,
        })
        
    def fetch_ohlcv_batch(self, symbol: str, timeframe: str = '1h', 
                         since: Optional[int] = None, limit: int = 1000) -> List[List]:
        """Fetch OHLCV data with error handling and rate limiting"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            logging.info(f"Fetched {len(ohlcv)} candles for {symbol}")
            return ohlcv
        except Exception as e:
            logging.error(f"Error fetching {symbol}: {e}")
            return []
            
    def fetch_historical_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Fetch historical data in chunks"""
        end_time = self.exchange.milliseconds()
        start_time = end_time - (days * 24 * 60 * 60 * 1000)
        
        all_ohlcv = []
        current_time = start_time
        
        while current_time < end_time:
            batch = self.fetch_ohlcv_batch(symbol, '1h', current_time, 1000)
            
            if not batch:
                break
                
            all_ohlcv.extend(batch)
            current_time = batch[-1][0] + (60 * 60 * 1000)  # Next hour
            
            # Rate limiting
            time.sleep(0.1)
            
        if all_ohlcv:
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df = df.drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
            
            logging.info(f"Total fetched for {symbol}: {len(df)} candles")
            return df
        
        return pd.DataFrame()
        
    def get_latest_candle(self, symbol: str) -> Optional[List]:
        """Get the most recent candle"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, '1h', None, 1)
            return ohlcv[0] if ohlcv else None
        except Exception as e:
            logging.error(f"Error fetching latest candle for {symbol}: {e}")
            return None
```

### Step 1.8: Database Operations with Data Integrity
```python
# src/database/operations.py
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Tuple
import json
import logging
from sqlalchemy import text
from .connection import DatabaseConnection

class DatabaseOperations:
    def __init__(self, database_url: str):
        self.db = DatabaseConnection(database_url)
        
    def save_ohlcv_data(self, symbol: str, timeframe: str, candles: List[List]) -> int:
        """Save OHLCV data with conflict handling and transaction support"""
        if not candles:
            return 0
            
        with self.db.get_session() as session:
            saved_count = 0
            
            try:
                # Begin transaction
                session.begin()
                
                for candle in candles:
                    timestamp = datetime.fromtimestamp(candle[0]/1000, tz=timezone.utc)
                    
                    # Use raw SQL for better performance with conflict handling
                    query = text("""
                        INSERT INTO ohlcv_data (symbol, timeframe, timestamp, open, high, low, close, volume)
                        VALUES (:symbol, :timeframe, :timestamp, :open, :high, :low, :close, :volume)
                        ON CONFLICT (symbol, timeframe, timestamp) DO NOTHING
                    """)
                    
                    result = session.execute(query, {
                        'symbol': symbol, 'timeframe': timeframe, 'timestamp': timestamp,
                        'open': candle[1], 'high': candle[2], 'low': candle[3], 
                        'close': candle[4], 'volume': candle[5]
                    })
                    
                    if result.rowcount > 0:
                        saved_count += 1
                        
                session.commit()
                logging.info(f"Saved {saved_count} new candles for {symbol}")
                
            except Exception as e:
                session.rollback()
                logging.error(f"Error saving OHLCV data: {e}")
                raise
                
            return saved_count
    
    def check_data_gaps(self, symbol: str, timeframe: str = '1h', 
                       hours_back: int = 168) -> List[Tuple[datetime, datetime]]:
        """Check for gaps in historical data"""
        with self.db.get_session() as session:
            # Get the expected range
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=hours_back)
            
            query = text("""
                WITH expected_hours AS (
                    SELECT generate_series(
                        :start_time::timestamp,
                        :end_time::timestamp,
                        '1 hour'::interval
                    ) AS expected_timestamp
                ),
                actual_data AS (
                    SELECT timestamp
                    FROM ohlcv_data
                    WHERE symbol = :symbol 
                    AND timeframe = :timeframe
                    AND timestamp >= :start_time
                    AND timestamp <= :end_time
                )
                SELECT 
                    expected_timestamp AS gap_start,
                    LEAD(expected_timestamp) OVER (ORDER BY expected_timestamp) AS gap_end
                FROM expected_hours
                WHERE expected_timestamp NOT IN (SELECT timestamp FROM actual_data)
                ORDER BY expected_timestamp
            """)
            
            result = session.execute(query, {
                'symbol': symbol,
                'timeframe': timeframe,
                'start_time': start_time,
                'end_time': end_time
            })
            
            gaps = []
            current_gap_start = None
            
            for row in result:
                if current_gap_start is None:
                    current_gap_start = row.gap_start
                elif row.gap_start - current_gap_start > timedelta(hours=1):
                    # End of continuous gap
                    gaps.append((current_gap_start, row.gap_start))
                    current_gap_start = row.gap_start
                    
            return gaps
    
    def get_last_complete_timestamp(self, symbol: str, timeframe: str = '1h') -> Optional[datetime]:
        """Get the timestamp of the last complete candle"""
        with self.db.get_session() as session:
            query = text("""
                SELECT MAX(timestamp) 
                FROM ohlcv_data 
                WHERE symbol = :symbol AND timeframe = :timeframe
            """)
            
            result = session.execute(query, {'symbol': symbol, 'timeframe': timeframe})
            last_timestamp = result.scalar()
            
            return last_timestamp
            
    def get_latest_candles(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> pd.DataFrame:
        """Get most recent candles"""
        with self.db.get_session() as session:
            query = """
                SELECT timestamp, open, high, low, close, volume
                FROM ohlcv_data 
                WHERE symbol = %s AND timeframe = %s
                ORDER BY timestamp DESC 
                LIMIT %s
            """
            
            result = session.execute(query, (symbol, timeframe, limit))
            rows = result.fetchall()
            
            if rows:
                df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['open'] = df['open'].astype(float)
                df['high'] = df['high'].astype(float)
                df['low'] = df['low'].astype(float)
                df['close'] = df['close'].astype(float)
                df['volume'] = df['volume'].astype(float)
                
                return df.sort_values('timestamp').reset_index(drop=True)
            
            return pd.DataFrame()
            
    def get_candle_count(self, symbol: str) -> int:
        """Get total number of candles for symbol"""
        with self.db.get_session() as session:
            query = "SELECT COUNT(*) FROM ohlcv_data WHERE symbol = %s"
            result = session.execute(query, (symbol,))
            count = result.fetchone()[0]
            return count
            
    def backfill_missing_data(self, symbol: str, gaps: List[Tuple[datetime, datetime]]) -> int:
        """Backfill missing data for identified gaps"""
        from src.data.fetcher import ExchangeDataFetcher
        fetcher = ExchangeDataFetcher()
        
        total_filled = 0
        
        for gap_start, gap_end in gaps:
            try:
                # Calculate time range
                start_ms = int(gap_start.timestamp() * 1000)
                end_ms = int(gap_end.timestamp() * 1000)
                
                # Fetch missing data
                logging.info(f"Backfilling {symbol} from {gap_start} to {gap_end}")
                missing_candles = fetcher.fetch_ohlcv_batch(symbol, '1h', start_ms, 1000)
                
                if missing_candles:
                    saved = self.save_ohlcv_data(symbol, '1h', missing_candles)
                    total_filled += saved
                    
            except Exception as e:
                logging.error(f"Error backfilling gap {gap_start} to {gap_end}: {e}")
                
        return total_filled
```

### Step 1.9: Data Recovery Manager
```python
# src/data/recovery.py
import logging
from datetime import datetime, timedelta
from typing import Dict, List

class DataRecoveryManager:
    def __init__(self, db_operations, data_fetcher):
        self.db = db_operations
        self.fetcher = data_fetcher
        
    def check_and_recover_all_symbols(self, symbols: List[str], hours_back: int = 168) -> Dict[str, int]:
        """Check data integrity and recover missing data for all symbols"""
        recovery_results = {}
        
        for symbol in symbols:
            logging.info(f"Checking data integrity for {symbol}")
            
            # Check for gaps
            gaps = self.db.check_data_gaps(symbol, '1h', hours_back)
            
            if gaps:
                logging.warning(f"Found {len(gaps)} data gaps for {symbol}")
                filled = self.db.backfill_missing_data(symbol, gaps)
                recovery_results[symbol] = filled
            else:
                logging.info(f"No data gaps found for {symbol}")
                recovery_results[symbol] = 0
                
        return recovery_results
    
    def ensure_minimum_data_coverage(self, symbol: str, min_hours: int = 168) -> bool:
        """Ensure minimum historical data is available"""
        last_timestamp = self.db.get_last_complete_timestamp(symbol)
        
        if not last_timestamp:
            # No data at all, fetch initial dataset
            logging.info(f"No data found for {symbol}, fetching initial dataset")
            df = self.fetcher.fetch_historical_data(symbol, days=min_hours//24 + 1)
            if len(df) > 0:
                candles = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].values.tolist()
                candles = [[int(row[0].timestamp() * 1000)] + list(row[1:]) for row in candles]
                self.db.save_ohlcv_data(symbol, '1h', candles)
                return True
            return False
            
        # Check if we have enough recent data
        hours_available = (datetime.now(timezone.utc) - last_timestamp).total_seconds() / 3600
        
        if hours_available > min_hours:
            logging.warning(f"Data for {symbol} is stale ({hours_available:.1f} hours old)")
            # Fetch recent data
            df = self.fetcher.fetch_historical_data(symbol, days=7)
            if len(df) > 0:
                candles = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].values.tolist()
                candles = [[int(row[0].timestamp() * 1000)] + list(row[1:]) for row in candles]
                self.db.save_ohlcv_data(symbol, '1h', candles)
                
        return True
```

**🔍 CHECKPOINT 1C TEST:**
```python
# test_data_pipeline.py
from src.data.fetcher import ExchangeDataFetcher
from src.database.operations import DatabaseOperations
from src.utils.config import Config

# Test data fetching
fetcher = ExchangeDataFetcher()
btc_data = fetcher.fetch_historical_data('BTC/USDT', days=2)  # Small test
print(f"✅ Fetched {len(btc_data)} BTC candles")

# Test database saving
config = Config()
db_ops = DatabaseOperations(config.database_url)

if len(btc_data) > 0:
    candles = btc_data[['timestamp', 'open', 'high', 'low', 'close', 'volume']].values.tolist()
    # Convert timestamp to milliseconds
    candles = [[int(row[0].timestamp() * 1000)] + list(row[1:]) for row in candles]
    
    saved = db_ops.save_ohlcv_data('BTC/USDT', '1h', candles)
    print(f"✅ Saved {saved} candles to database")
    
    # Test retrieval
    retrieved = db_ops.get_latest_candles('BTC/USDT', limit=10)
    print(f"✅ Retrieved {len(retrieved)} candles from database")
    
    # Verify data integrity
    if len(retrieved) > 0:
        print(f"✅ Data integrity check passed")
        print(f"   Latest candle: {retrieved.iloc[-1]['timestamp']}")
        print(f"   Price range: ${retrieved['low'].min():.2f} - ${retrieved['high'].max():.2f}")
```

---

## Phase 1 Success Criteria

After completing Phase 1, you should have:

✅ **Working project structure** with proper configuration  
✅ **PostgreSQL database** running with TimescaleDB  
✅ **Data fetcher** that can pull crypto data from exchanges  
✅ **Database operations** that store and retrieve OHLCV data  
✅ **All tests passing** for basic functionality  

**Next Step**: Move to Phase 2 to build the ML prediction engine that will use this data foundation.
