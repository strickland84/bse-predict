"""Exchange data fetching using CCXT."""
import ccxt
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict
import time
import logging

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ExchangeDataFetcher:
    """Fetches cryptocurrency data from exchanges using CCXT."""
    
    def __init__(self, exchange_name: str = None):
        """Initialize exchange data fetcher.
        
        Args:
            exchange_name: Exchange name (e.g., 'binance', 'coinbasepro')
        """
        from src.utils.config import config
        exchange_name = exchange_name or config.exchange_name
        
        self.exchange = getattr(ccxt, exchange_name)({
            'apiKey': '',  # Not needed for public data
            'secret': '',
            'timeout': 30000,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',  # Use spot markets
            }
        })
        
    def fetch_ohlcv_batch(self, symbol: str, timeframe: str = '1h', 
                         since: Optional[int] = None, limit: int = 1000) -> List[List]:
        """Fetch OHLCV data with error handling and rate limiting.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1h', '4h', '1d')
            since: Start timestamp in milliseconds
            limit: Maximum number of candles
            
        Returns:
            List of OHLCV data [timestamp, open, high, low, close, volume]
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            logger.info(f"Fetched {len(ohlcv)} candles for {symbol}")
            return ohlcv
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return []
            
    def fetch_historical_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Fetch historical data in chunks, optimized for maximum data retrieval.
        
        Args:
            symbol: Trading pair symbol
            days: Number of days to fetch (can handle 3+ years)
            
        Returns:
            DataFrame with OHLCV data
        """
        end_time = self.exchange.milliseconds()
        start_time = end_time - (days * 24 * 60 * 60 * 1000)
        
        all_ohlcv = []
        current_time = start_time
        attempts = 0
        max_attempts = days * 2  # Allow more attempts for longer periods
        consecutive_empty = 0
        
        logger.info(f"🎯 Starting fetch for {symbol}: {days} days ({days * 24} hours)")
        
        while current_time < end_time and attempts < max_attempts:
            # Use maximum limit (1000) for Binance
            batch = self.fetch_ohlcv_batch(symbol, '1h', current_time, 1000)
            attempts += 1
            
            if not batch:
                consecutive_empty += 1
                # If we get 3 empty responses in a row, data might not exist that far back
                if consecutive_empty >= 3:
                    logger.warning(f"No data available before {pd.to_datetime(current_time, unit='ms')}")
                    # Jump forward significantly to find where data starts
                    current_time += (30 * 24 * 60 * 60 * 1000)  # Jump 30 days
                else:
                    current_time += (24 * 60 * 60 * 1000)  # Jump 1 day
                continue
            
            consecutive_empty = 0  # Reset counter on successful fetch
            all_ohlcv.extend(batch)
            
            # Move to next batch starting point
            # Since we got 1000 candles (max), jump exactly 1000 hours ahead
            last_timestamp = batch[-1][0]
            current_time = last_timestamp + (60 * 60 * 1000)  # Next hour after last candle
            
            # Adaptive rate limiting based on request size
            if days > 365:
                time.sleep(0.2)  # Slower for very large requests
            elif days > 90:
                time.sleep(0.1)  # Medium delay
            else:
                time.sleep(0.05)  # Fast for smaller requests
            
            # Progress indicator - more frequent updates for large fetches
            if len(all_ohlcv) % 2000 == 0 and len(all_ohlcv) > 0:
                progress_days = len(all_ohlcv) // 24
                logger.info(f"   Progress: {len(all_ohlcv)} candles ({progress_days} days) fetched for {symbol}...")
            
        if all_ohlcv:
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df = df.drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
            
            actual_days = len(df) // 24
            date_range = f"{df['timestamp'].iloc[0].date()} to {df['timestamp'].iloc[-1].date()}"
            logger.info(f"✅ Total fetched for {symbol}: {len(df)} candles ({actual_days} days)")
            logger.info(f"   Date range: {date_range}")
            return df
        
        return pd.DataFrame()
        
    def get_latest_candle(self, symbol: str) -> Optional[List]:
        """Get the most recent candle.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Latest OHLCV candle or None if error
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, '1h', None, 1)
            return ohlcv[0] if ohlcv else None
        except Exception as e:
            logger.error(f"Error fetching latest candle for {symbol}: {e}")
            return None
            
    def fetch_latest_candle(self, symbol: str) -> Optional[dict]:
        """Get the most recent candle with additional metadata.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dictionary with candle data and metadata
        """
        try:
            candle = self.get_latest_candle(symbol)
            if candle:
                return {
                    'symbol': symbol,
                    'timestamp': candle[0],
                    'open': candle[1],
                    'high': candle[2],
                    'low': candle[3],
                    'close': candle[4],
                    'volume': candle[5],
                    'human_time': pd.to_datetime(candle[0], unit='ms', utc=True).strftime('%Y-%m-%d %H:%M UTC')
                }
            return None
        except Exception as e:
            logger.error(f"Error fetching latest candle for {symbol}: {e}")
            return None
            
    def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol exists on exchange.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            True if symbol exists, False otherwise
        """
        try:
            markets = self.exchange.load_markets()
            return symbol in markets
        except Exception as e:
            logger.error(f"Error validating symbol {symbol}: {e}")
            return False
            
    def get_exchange_info(self) -> Dict:
        """Get exchange information.
        
        Returns:
            Dictionary with exchange information
        """
        try:
            self.exchange.load_markets()
            return {
                'name': self.exchange.name,
                'symbols': list(self.exchange.symbols),
                'timeframes': list(self.exchange.timeframes.keys()),
                'rate_limit': self.exchange.rateLimit
            }
        except Exception as e:
            logger.error(f"Error getting exchange info: {e}")
            return {}
            
    def get_symbol_info(self, symbol: str) -> Dict:
        """Get detailed symbol information.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dictionary with symbol information
        """
        try:
            markets = self.exchange.load_markets()
            if symbol in markets:
                market = markets[symbol]
                return {
                    'symbol': symbol,
                    'base': market['base'],
                    'quote': market['quote'],
                    'precision': market['precision'],
                    'limits': market['limits'],
                    'active': market['active']
                }
            return {}
        except Exception as e:
            logger.error(f"Error getting symbol info for {symbol}: {e}")
            return {}
