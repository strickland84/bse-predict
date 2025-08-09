"""Database operations for BSE Predict."""
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Tuple
import json
import logging
from sqlalchemy import text
from pathlib import Path

from src.database.connection import DatabaseConnection
from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Global database operations instance
db_connection = DatabaseConnection(config.database_url)
db_ops = db_connection


class DatabaseOperations:
    """Enhanced database operations with data integrity and recovery."""
    
    def __init__(self, database_url: str):
        """Initialize database operations.
        
        Args:
            database_url: PostgreSQL connection string
        """
        self.db = DatabaseConnection(database_url)
        
    def save_ohlcv_data(self, symbol: str, timeframe: str, candles: List[List]) -> int:
        """Save OHLCV data with conflict handling and transaction support.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe (e.g., '1h')
            candles: List of [timestamp, open, high, low, close, volume] lists
            
        Returns:
            Number of new candles saved
        """
        if not candles:
            return 0
            
        with self.db.get_session() as session:
            saved_count = 0
            
            try:
                session.begin()
                
                for candle in candles:
                    timestamp = datetime.fromtimestamp(candle[0]/1000, tz=timezone.utc)
                    
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
                logger.info(f"Saved {saved_count} new candles for {symbol}")
                
            except Exception as e:
                session.rollback()
                logger.error(f"Error saving OHLCV data: {e}")
                raise
                
            return saved_count
    
    def check_data_gaps(self, symbol: str, timeframe: str = '1h', 
                       hours_back: int = 168) -> List[Tuple[datetime, datetime]]:
        """Check for gaps in historical data.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe to check
            hours_back: How many hours to look back
            
        Returns:
            List of (gap_start, gap_end) tuples
        """
        with self.db.get_session() as session:
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=hours_back)
            
            # Find gaps by looking at consecutive candles
            query = text("""
                WITH candle_data AS (
                    SELECT 
                        timestamp,
                        LAG(timestamp) OVER (ORDER BY timestamp) as prev_timestamp
                    FROM ohlcv_data
                    WHERE symbol = :symbol 
                    AND timeframe = :timeframe
                    AND timestamp >= CAST(:start_time AS timestamp)
                    AND timestamp <= CAST(:end_time AS timestamp)
                    ORDER BY timestamp
                )
                SELECT 
                    prev_timestamp as gap_start,
                    timestamp as gap_end,
                    EXTRACT(EPOCH FROM (timestamp - prev_timestamp)) / 3600 as gap_hours
                FROM candle_data
                WHERE timestamp - prev_timestamp > INTERVAL '1 hour 5 minutes'
                ORDER BY prev_timestamp
            """)
            
            result = session.execute(query, {
                'symbol': symbol,
                'timeframe': timeframe,
                'start_time': start_time,
                'end_time': end_time
            })
            
            gaps = []
            
            for row in result:
                if row.gap_start and row.gap_end:
                    gaps.append((row.gap_start, row.gap_end))
                    
            return gaps
    
    def get_last_complete_timestamp(self, symbol: str, timeframe: str = '1h') -> Optional[datetime]:
        """Get the timestamp of the last complete candle.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe to check
            
        Returns:
            Last timestamp or None if no data
        """
        with self.db.get_session() as session:
            query = text("""
                SELECT MAX(timestamp) 
                FROM ohlcv_data 
                WHERE symbol = :symbol AND timeframe = :timeframe
            """)
            
            result = session.execute(query, {'symbol': symbol, 'timeframe': timeframe})
            last_timestamp = result.scalar()
            
            return last_timestamp
            
    def get_latest_candles(self, symbol: str, timeframe: str = '1h', limit: int = 100, include_futures: bool = True) -> pd.DataFrame:
        """Get most recent candles with optional futures data.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            limit: Maximum number of candles
            include_futures: Whether to join futures data
            
        Returns:
            DataFrame with OHLCV data and optionally futures data
        """
        with self.db.get_session() as session:
            # Check if futures_data table exists before trying to join
            futures_table_exists = False
            if include_futures:
                try:
                    check_query = text("""
                        SELECT EXISTS (
                            SELECT 1 FROM information_schema.tables 
                            WHERE table_name = 'futures_data'
                        )
                    """)
                    futures_table_exists = session.execute(check_query).scalar()
                except:
                    futures_table_exists = False
            
            if include_futures and futures_table_exists:
                # Join OHLCV with futures data
                query = text("""
                    SELECT 
                        o.timestamp, o.open, o.high, o.low, o.close, o.volume,
                        f.open_interest, f.open_interest_value, f.funding_rate,
                        f.top_trader_ratio, f.taker_buy_sell_ratio
                    FROM ohlcv_data o
                    LEFT JOIN futures_data f 
                        ON o.symbol = f.symbol 
                        AND o.timestamp = f.timestamp
                    WHERE o.symbol = :symbol AND o.timeframe = :timeframe
                    ORDER BY o.timestamp DESC 
                    LIMIT :limit
                """)
            else:
                query = text("""
                    SELECT timestamp, open, high, low, close, volume
                    FROM ohlcv_data 
                    WHERE symbol = :symbol AND timeframe = :timeframe
                    ORDER BY timestamp DESC 
                    LIMIT :limit
                """)
            
            result = session.execute(query, {"symbol": symbol, "timeframe": timeframe, "limit": limit})
            rows = result.fetchall()
            
            if rows:
                if include_futures:
                    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                              'open_interest', 'open_interest_value', 'funding_rate',
                              'top_trader_ratio', 'taker_buy_sell_ratio']
                else:
                    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    
                df = pd.DataFrame(rows, columns=columns)
                
                # Convert numeric columns to float
                numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                if include_futures:
                    numeric_cols.extend(['open_interest', 'open_interest_value', 'funding_rate',
                                       'top_trader_ratio', 'taker_buy_sell_ratio'])
                
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                return df.sort_values('timestamp').reset_index(drop=True)
            
            return pd.DataFrame()
    
    def get_candles_from_timestamp(self, symbol: str, start_timestamp: datetime, 
                                   timeframe: str = '1h', limit: int = 100,
                                   include_futures: bool = False) -> pd.DataFrame:
        """Get candles starting from a specific timestamp.
        
        Args:
            symbol: Trading pair symbol
            start_timestamp: Starting timestamp (inclusive)
            timeframe: Timeframe
            limit: Maximum number of candles to return
            include_futures: Whether to join futures data
            
        Returns:
            DataFrame with OHLCV data starting from the given timestamp
        """
        with self.db.get_session() as session:
            query = text("""
                SELECT timestamp, open, high, low, close, volume
                FROM ohlcv_data 
                WHERE symbol = :symbol 
                    AND timeframe = :timeframe
                    AND timestamp >= :start_timestamp
                ORDER BY timestamp ASC
                LIMIT :limit
            """)
            
            result = session.execute(query, {
                "symbol": symbol, 
                "timeframe": timeframe, 
                "start_timestamp": start_timestamp,
                "limit": limit
            })
            rows = result.fetchall()
            
            if rows:
                columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                df = pd.DataFrame(rows, columns=columns)
                
                # Convert numeric columns to float
                numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                return df
            
            return pd.DataFrame()
            
    def get_candle_count(self, symbol: str) -> int:
        """Get total number of candles for symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Number of candles
        """
        with self.db.get_session() as session:
            query = text("SELECT COUNT(*) FROM ohlcv_data WHERE symbol = :symbol")
            result = session.execute(query, {'symbol': symbol})
            count = result.fetchone()[0]
            return count
    
    def get_oldest_candle(self, symbol: str) -> Optional[Dict]:
        """Get the oldest candle for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dictionary with oldest candle data or None if no data
        """
        with self.db.get_session() as session:
            query = text("""
                SELECT timestamp, open, high, low, close, volume 
                FROM ohlcv_data 
                WHERE symbol = :symbol 
                ORDER BY timestamp ASC 
                LIMIT 1
            """)
            result = session.execute(query, {'symbol': symbol}).fetchone()
            
            if result:
                return {
                    'timestamp': result.timestamp,
                    'open': float(result.open),
                    'high': float(result.high),
                    'low': float(result.low),
                    'close': float(result.close),
                    'volume': float(result.volume)
                }
            return None
            
    def backfill_missing_data(self, symbol: str, gaps: List[Tuple[datetime, datetime]]) -> int:
        """Backfill missing data for identified gaps.
        
        Args:
            symbol: Trading pair symbol
            gaps: List of (gap_start, gap_end) tuples
            
        Returns:
            Number of candles filled
        """
        from src.data.fetcher import ExchangeDataFetcher
        
        fetcher = ExchangeDataFetcher()
        total_filled = 0
        
        for gap_start, gap_end in gaps:
            try:
                # Calculate time range
                start_ms = int(gap_start.timestamp() * 1000)
                end_ms = int(gap_end.timestamp() * 1000)
                
                # Fetch missing data
                logger.debug(f"Backfilling {symbol} from {gap_start} to {gap_end}")  # Changed to debug level
                missing_candles = fetcher.fetch_ohlcv_batch(symbol, '1h', start_ms, 1000)
                
                if missing_candles:
                    saved = self.save_ohlcv_data(symbol, '1h', missing_candles)
                    total_filled += saved
                    
            except Exception as e:
                logger.error(f"Error backfilling gap {gap_start} to {gap_end}: {e}")
                
        return total_filled
    
    def create_futures_table(self):
        """Create futures_data table if it doesn't exist."""
        # First, create the table
        with self.db.get_session() as session:
            try:
                query = text("""
                    CREATE TABLE IF NOT EXISTS futures_data (
                        id SERIAL,
                        symbol TEXT NOT NULL,
                        timestamp TIMESTAMPTZ NOT NULL,
                        open_interest DECIMAL(20, 8),
                        open_interest_value DECIMAL(20, 8),
                        funding_rate DECIMAL(10, 8),
                        mark_price DECIMAL(20, 8),
                        top_trader_ratio DECIMAL(10, 4),
                        taker_buy_sell_ratio DECIMAL(10, 4),
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        PRIMARY KEY (id, timestamp),
                        UNIQUE(symbol, timestamp)
                    );
                """)
                session.execute(query)
                session.commit()
                logger.info("Futures data table created")
            except Exception as e:
                session.rollback()
                if "already exists" not in str(e):
                    logger.error(f"Error creating futures table: {e}")
                    raise
        
        # Then try to create TimescaleDB hypertable in a separate transaction
        with self.db.get_session() as session:
            try:
                session.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"))
                
                # Check if table is already a hypertable
                is_hypertable = session.execute(text("""
                    SELECT EXISTS (
                        SELECT 1 FROM _timescaledb_catalog.hypertable 
                        WHERE table_name = 'futures_data'
                    );
                """)).scalar()
                
                if not is_hypertable:
                    # Check if table has data
                    has_data = session.execute(text("SELECT EXISTS (SELECT 1 FROM futures_data LIMIT 1);")).scalar()
                    
                    if has_data:
                        logger.info("Futures table has existing data, migrating to hypertable...")
                        session.execute(text("SELECT create_hypertable('futures_data', 'timestamp', migrate_data => TRUE, if_not_exists => TRUE);"))
                    else:
                        session.execute(text("SELECT create_hypertable('futures_data', 'timestamp', if_not_exists => TRUE);"))
                    
                    logger.info("TimescaleDB hypertable created for futures_data")
                else:
                    logger.debug("Futures table is already a hypertable")
                    
                session.commit()
            except Exception as e:
                session.rollback()
                # TimescaleDB not available or error, continue with regular table
                logger.debug(f"TimescaleDB not configured for futures_data: {e}")
        
        # Finally, create the index in another transaction
        with self.db.get_session() as session:
            try:
                session.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_futures_symbol_time 
                    ON futures_data (symbol, timestamp DESC);
                """))
                session.commit()
                logger.info("Futures data index created")
            except Exception as e:
                session.rollback()
                if "already exists" not in str(e):
                    logger.error(f"Error creating index: {e}")
                    # Don't raise, the table is still usable without the index
    
    def save_futures_data(self, symbol: str, timestamp: datetime, 
                         open_interest: Optional[float] = None,
                         open_interest_value: Optional[float] = None,
                         funding_rate: Optional[float] = None,
                         mark_price: Optional[float] = None,
                         top_trader_ratio: Optional[float] = None,
                         taker_buy_sell_ratio: Optional[float] = None) -> bool:
        """Save futures data point.
        
        Args:
            symbol: Trading pair symbol
            timestamp: Data timestamp
            open_interest: Open interest in base currency
            open_interest_value: Open interest in USDT
            funding_rate: Current funding rate
            mark_price: Mark price
            top_trader_ratio: Top trader long/short ratio
            taker_buy_sell_ratio: Taker buy/sell volume ratio
            
        Returns:
            True if saved successfully
        """
        with self.db.get_session() as session:
            try:
                query = text("""
                    INSERT INTO futures_data 
                    (symbol, timestamp, open_interest, open_interest_value, 
                     funding_rate, mark_price, top_trader_ratio, taker_buy_sell_ratio)
                    VALUES (:symbol, :timestamp, :open_interest, :open_interest_value,
                            :funding_rate, :mark_price, :top_trader_ratio, :taker_buy_sell_ratio)
                    ON CONFLICT (symbol, timestamp) DO UPDATE SET
                        open_interest = COALESCE(EXCLUDED.open_interest, futures_data.open_interest),
                        open_interest_value = COALESCE(EXCLUDED.open_interest_value, futures_data.open_interest_value),
                        funding_rate = COALESCE(EXCLUDED.funding_rate, futures_data.funding_rate),
                        mark_price = COALESCE(EXCLUDED.mark_price, futures_data.mark_price),
                        top_trader_ratio = COALESCE(EXCLUDED.top_trader_ratio, futures_data.top_trader_ratio),
                        taker_buy_sell_ratio = COALESCE(EXCLUDED.taker_buy_sell_ratio, futures_data.taker_buy_sell_ratio)
                """)
                
                # Convert numpy types to Python native types
                import numpy as np
                
                def to_python_type(value):
                    if value is None:
                        return None
                    if isinstance(value, (np.integer, np.floating)):
                        return float(value)
                    return value
                
                session.execute(query, {
                    'symbol': symbol,
                    'timestamp': timestamp,
                    'open_interest': to_python_type(open_interest),
                    'open_interest_value': to_python_type(open_interest_value),
                    'funding_rate': to_python_type(funding_rate),
                    'mark_price': to_python_type(mark_price),
                    'top_trader_ratio': to_python_type(top_trader_ratio),
                    'taker_buy_sell_ratio': to_python_type(taker_buy_sell_ratio)
                })
                
                session.commit()
                return True
                
            except Exception as e:
                session.rollback()
                logger.error(f"Error saving futures data: {e}")
                return False
    
    def get_latest_futures_data(self, symbol: str, hours_back: int = 24) -> pd.DataFrame:
        """Get recent futures data for a symbol.
        
        Args:
            symbol: Trading pair symbol
            hours_back: How many hours of data to retrieve
            
        Returns:
            DataFrame with futures data
        """
        with self.db.get_session() as session:
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=hours_back)
            
            query = text("""
                SELECT timestamp, open_interest, open_interest_value, 
                       funding_rate, mark_price, top_trader_ratio, taker_buy_sell_ratio
                FROM futures_data
                WHERE symbol = :symbol 
                AND timestamp >= :start_time
                AND timestamp <= :end_time
                ORDER BY timestamp
            """)
            
            result = session.execute(query, {
                'symbol': symbol,
                'start_time': start_time,
                'end_time': end_time
            })
            
            rows = result.fetchall()
            if rows:
                df = pd.DataFrame(rows, columns=[
                    'timestamp', 'open_interest', 'open_interest_value',
                    'funding_rate', 'mark_price', 'top_trader_ratio', 'taker_buy_sell_ratio'
                ])
                
                # Convert to float
                for col in ['open_interest', 'open_interest_value', 'funding_rate', 
                           'mark_price', 'top_trader_ratio', 'taker_buy_sell_ratio']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                return df
            
            return pd.DataFrame()
    
    def save_prediction(self, symbol: str, target_pct: float, timestamp: datetime,
                       model_name: str, prediction_class: int, probability: float,
                       confidence: float, features_used: Dict) -> Optional[int]:
        """Save a prediction to the database.
        
        Args:
            symbol: Trading pair symbol
            target_pct: Target percentage
            timestamp: Prediction timestamp
            model_name: Model name
            prediction_class: 0 for DOWN, 1 for UP
            probability: Prediction probability
            confidence: Confidence score
            features_used: Features dictionary
            
        Returns:
            Prediction ID if successful
        """
        from src.database.prediction_tracker import prediction_tracker
        return prediction_tracker.save_prediction(
            symbol, target_pct, timestamp, model_name, prediction_class,
            probability, confidence, features_used
        )
    
    def save_predictions_batch(self, predictions: List[Dict]) -> List[int]:
        """Save multiple predictions in batch.
        
        Args:
            predictions: List of prediction dictionaries
            
        Returns:
            List of prediction IDs
        """
        from src.database.prediction_tracker import prediction_tracker
        return prediction_tracker.save_predictions_batch(predictions)
    
    def get_pending_predictions(self, window_hours: int = 24) -> pd.DataFrame:
        """Get predictions that need outcome tracking.
        
        Args:
            window_hours: Hours to look back
            
        Returns:
            DataFrame with pending predictions
        """
        from src.database.prediction_tracker import prediction_tracker
        return prediction_tracker.get_pending_predictions(window_hours)
    
    def update_prediction_outcome(self, prediction_id: int, actual_outcome: int,
                                target_hit_timestamp: Optional[datetime] = None,
                                time_to_target_hours: Optional[float] = None,
                                max_favorable_move: Optional[float] = None,
                                max_adverse_move: Optional[float] = None) -> bool:
        """Update prediction outcome.
        
        Args:
            prediction_id: Prediction ID
            actual_outcome: 0 for DOWN, 1 for UP
            target_hit_timestamp: When target was hit
            time_to_target_hours: Hours to target
            max_favorable_move: Max favorable move
            max_adverse_move: Max adverse move
            
        Returns:
            True if successful
        """
        from src.database.prediction_tracker import prediction_tracker
        return prediction_tracker.update_prediction_outcome(
            prediction_id, actual_outcome, target_hit_timestamp,
            time_to_target_hours, max_favorable_move, max_adverse_move
        )
    
    def get_prediction_performance(self, symbol: Optional[str] = None,
                                 target_pct: Optional[float] = None,
                                 days_back: int = 7) -> pd.DataFrame:
        """Get prediction performance statistics.
        
        Args:
            symbol: Filter by symbol
            target_pct: Filter by target
            days_back: Days to analyze
            
        Returns:
            DataFrame with performance metrics
        """
        from src.database.prediction_tracker import prediction_tracker
        return prediction_tracker.get_prediction_performance(
            symbol, target_pct, days_back
        )


# Create global instance
db_ops = DatabaseOperations(config.database_url)
