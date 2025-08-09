"""Data recovery and integrity management."""
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple
from pathlib import Path

from src.data.fetcher import ExchangeDataFetcher
from src.database.operations import DatabaseOperations
from src.utils.logger import get_logger
from src.utils.config import config

logger = get_logger(__name__)


class DataRecoveryManager:
    """Manages data integrity and recovery operations."""
    
    def __init__(self):
        """Initialize data recovery manager."""
        self.fetcher = ExchangeDataFetcher()
        self.db_ops = DatabaseOperations(config.database_url)
        self.futures_fetcher = None  # Lazy initialization
        
    def check_and_recover_all_symbols(self, symbols: List[str] = None, 
                                    hours_back: int = 168) -> Dict[str, int]:
        """Check data integrity and recover missing data for all symbols.
        
        Args:
            symbols: List of symbols to check. If None, uses config assets.
            hours_back: How many hours to look back for gaps.
            
        Returns:
            Dictionary mapping symbols to number of candles filled.
        """
        if symbols is None:
            symbols = config.assets
            
        recovery_results = {}
        
        for symbol in symbols:
            logger.info(f"🔍 Checking data integrity for {symbol}")
            
            # Check for gaps
            gaps = self.db_ops.check_data_gaps(symbol, '1h', hours_back)
            
            if gaps:
                logger.warning(f"Found {len(gaps)} data gaps for {symbol}")
                filled = self.db_ops.backfill_missing_data(symbol, gaps)
                recovery_results[symbol] = filled
            else:
                logger.info(f"No data gaps found for {symbol}")
                recovery_results[symbol] = 0
                
        return recovery_results
    
    def ensure_minimum_data_coverage(self, symbol: str, min_hours: int = 168) -> bool:
        """Ensure minimum historical data is available.
        
        Args:
            symbol: Trading pair symbol
            min_hours: Minimum hours of data required
            
        Returns:
            True if data coverage is sufficient, False otherwise
        """
        last_timestamp = self.db_ops.get_last_complete_timestamp(symbol)
        
        if not last_timestamp:
            # No data at all, fetch initial dataset
            logger.info(f"No data found for {symbol}, fetching initial dataset")
            df = self.fetcher.fetch_historical_data(symbol, days=min_hours//24 + 1)
            if len(df) > 0:
                candles = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].values.tolist()
                candles = [[int(row[0].timestamp() * 1000)] + list(row[1:]) for row in candles]
                self.db_ops.save_ohlcv_data(symbol, '1h', candles)
                return True
            return False
            
        # Check if we have enough recent data
        hours_available = (datetime.now(timezone.utc) - last_timestamp).total_seconds() / 3600
        
        if hours_available > min_hours:
            logger.warning(f"Data for {symbol} is stale ({hours_available:.1f} hours old)")
            # Fetch recent data
            df = self.fetcher.fetch_historical_data(symbol, days=7)
            if len(df) > 0:
                candles = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].values.tolist()
                candles = [[int(row[0].timestamp() * 1000)] + list(row[1:]) for row in candles]
                self.db_ops.save_ohlcv_data(symbol, '1h', candles)
                
        return True
    
    def ensure_all_symbols_coverage(self, min_candles: int = 1000, target_years: int = 3) -> Dict[str, int]:
        """Ensure all symbols have maximum possible data coverage (3+ years if available).
        
        Args:
            min_candles: Minimum number of candles required per symbol
            target_years: Target years of historical data to fetch (default 3)
            
        Returns:
            Dictionary mapping symbols to number of candles after operation
        """
        results = {}
        target_candles = target_years * 365 * 24  # Target in hours
        
        # First, show current status for all symbols
        print("\n📊 Data Status Check:")
        print("   " + "-" * 45)
        
        for symbol in config.assets:
            current_count = self.db_ops.get_candle_count(symbol)
            oldest = self.db_ops.get_oldest_candle(symbol)
            
            if oldest:
                days_of_data = (datetime.now(timezone.utc) - oldest['timestamp']).days
                years_of_data = days_of_data / 365.0
                
                # Check if we need more data
                if current_count < target_candles and days_of_data < (target_years * 365):
                    print(f"   ⚠️  {symbol}: {current_count} candles ({days_of_data} days / {years_of_data:.1f} years)")
                    print(f"      → Will attempt to fetch {target_years} years of data...")
                    
                    # Fetch maximum historical data (3+ years)
                    target_days = target_years * 365
                    print(f"      Fetching up to {target_days} days...")
                    
                    df = self.fetcher.fetch_historical_data(symbol, days=target_days)
                    
                    if len(df) > 0:
                        candles = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].values.tolist()
                        candles = [[int(row[0].timestamp() * 1000)] + list(row[1:]) for row in candles]
                        saved = self.db_ops.save_ohlcv_data(symbol, '1h', candles)
                        
                        if saved > 0:
                            # Get updated stats
                            new_count = self.db_ops.get_candle_count(symbol)
                            new_oldest = self.db_ops.get_oldest_candle(symbol)
                            if new_oldest:
                                new_days = (datetime.now(timezone.utc) - new_oldest['timestamp']).days
                                new_years = new_days / 365.0
                                print(f"      ✅ Added {saved} candles. Total: {new_count} ({new_days} days / {new_years:.1f} years)")
                else:
                    print(f"   ✅ {symbol}: {current_count} candles ({days_of_data} days / {years_of_data:.1f} years)")
            else:
                # No data at all, fetch maximum
                print(f"   ❌ {symbol}: No data found")
                print(f"      → Fetching up to {target_years} years of data...")
                
                target_days = target_years * 365
                df = self.fetcher.fetch_historical_data(symbol, days=target_days)
                
                if len(df) > 0:
                    candles = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].values.tolist()
                    candles = [[int(row[0].timestamp() * 1000)] + list(row[1:]) for row in candles]
                    saved = self.db_ops.save_ohlcv_data(symbol, '1h', candles)
                    
                    if saved > 0:
                        new_count = self.db_ops.get_candle_count(symbol)
                        new_oldest = self.db_ops.get_oldest_candle(symbol)
                        if new_oldest:
                            new_days = (datetime.now(timezone.utc) - new_oldest['timestamp']).days
                            new_years = new_days / 365.0
                            print(f"      ✅ Added {saved} candles. Total: {new_count} ({new_days} days / {new_years:.1f} years)")
            
            # Store final count
            results[symbol] = self.db_ops.get_candle_count(symbol)
        
        # Show final summary
        print("\n📊 Final Data Coverage:")
        print("   " + "-" * 45)
        for symbol, count in results.items():
            oldest = self.db_ops.get_oldest_candle(symbol)
            if oldest:
                days = (datetime.now(timezone.utc) - oldest['timestamp']).days
                years = days / 365.0
                date_range = f"{oldest['timestamp'].date()} to now"
                print(f"   ✅ {symbol}: {count} candles ({days} days / {years:.1f} years)")
                print(f"      Range: {date_range}")
            
        return results
        
    def get_data_health_report(self, symbols: List[str] = None) -> Dict[str, Dict]:
        """Get comprehensive data health report for all symbols.
        
        Args:
            symbols: List of symbols to check
            
        Returns:
            Dictionary with data health metrics
        """
        if symbols is None:
            symbols = config.assets
            
        report = {}
        
        for symbol in symbols:
            # Get basic stats
            total_candles = self.db_ops.get_candle_count(symbol)
            latest_candles = self.db_ops.get_latest_candles(symbol, limit=1)
            
            if latest_candles.empty:
                report[symbol] = {
                    'status': 'NO_DATA',
                    'total_candles': 0,
                    'latest_timestamp': None,
                    'hours_old': None,
                    'gaps': []
                }
                continue
                
            latest_timestamp = latest_candles.iloc[0]['timestamp']
            hours_old = (datetime.now(timezone.utc) - latest_timestamp).total_seconds() / 3600
            
            # Check for gaps
            gaps = self.db_ops.check_data_gaps(symbol, '1h', hours_back=168)
            
            report[symbol] = {
                'status': 'HEALTHY' if hours_old < 2 and len(gaps) == 0 else 'NEEDS_ATTENTION',
                'total_candles': total_candles,
                'latest_timestamp': latest_timestamp.isoformat(),
                'hours_old': round(hours_old, 1),
                'gaps_count': len(gaps),
                'gaps': [(gap[0].isoformat(), gap[1].isoformat()) for gap in gaps[:5]]  # Show first 5
            }
            
        return report
        
    def backfill_historical_data(self, symbols: List[str] = None, days: int = 180) -> Dict[str, int]:
        """Backfill historical data for all symbols, optimized for maximum data.
        
        Args:
            symbols: List of symbols to backfill
            days: Number of days to backfill (supports 3+ years)
            
        Returns:
            Dictionary mapping symbols to number of candles added
        """
        if symbols is None:
            symbols = config.assets
            
        results = {}
        
        for symbol in symbols:
            logger.info(f"📈 Backfilling {days} days of data for {symbol}")
            
            # Check current data
            current_count = self.db_ops.get_candle_count(symbol)
            oldest_candle = self.db_ops.get_oldest_candle(symbol)
            
            if oldest_candle:
                existing_days = (datetime.now(timezone.utc) - oldest_candle['timestamp']).days
                logger.info(f"   Current: {current_count} candles ({existing_days} days)")
            
            # Fetch historical data with optimized fetcher
            df = self.fetcher.fetch_historical_data(symbol, days=days)
            
            if df.empty:
                logger.warning(f"❌ No data fetched for {symbol}")
                results[symbol] = 0
                continue
                
            # Convert and save
            candles = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].values.tolist()
            candles = [[int(row[0].timestamp() * 1000)] + list(row[1:]) for row in candles]
            
            saved = self.db_ops.save_ohlcv_data(symbol, '1h', candles)
            results[symbol] = saved
            
            # Get updated stats
            new_count = self.db_ops.get_candle_count(symbol)
            new_oldest = self.db_ops.get_oldest_candle(symbol)
            if new_oldest:
                total_days = (datetime.now(timezone.utc) - new_oldest['timestamp']).days
                logger.info(f"✅ Added {saved} new candles for {symbol}")
                logger.info(f"   Total: {new_count} candles ({total_days} days)")
            
        return results
    
    def backfill_maximum_data(self, symbols: List[str] = None) -> Dict[str, Dict]:
        """Attempt to backfill maximum possible historical data (3+ years).
        
        Args:
            symbols: List of symbols to backfill
            
        Returns:
            Dictionary with results for each symbol
        """
        if symbols is None:
            symbols = config.assets
        
        results = {}
        target_days = 365 * 3  # Try for 3 years
        
        logger.info(f"🚀 Starting maximum data backfill (target: {target_days} days)")
        
        for symbol in symbols:
            logger.info(f"\n📊 Processing {symbol}...")
            
            # Check current state
            current_count = self.db_ops.get_candle_count(symbol)
            oldest = self.db_ops.get_oldest_candle(symbol)
            
            if oldest:
                current_days = (datetime.now(timezone.utc) - oldest['timestamp']).days
                logger.info(f"   Current: {current_count} candles ({current_days} days)")
                
                if current_days >= target_days:
                    logger.info(f"   ✅ Already has {current_days} days of data")
                    results[symbol] = {
                        'status': 'sufficient',
                        'candles': current_count,
                        'days': current_days,
                        'added': 0
                    }
                    continue
            
            # Try to fetch maximum data
            logger.info(f"   Fetching up to {target_days} days...")
            df = self.fetcher.fetch_historical_data(symbol, days=target_days)
            
            if df.empty:
                logger.warning(f"   ❌ No data fetched")
                results[symbol] = {
                    'status': 'failed',
                    'candles': current_count,
                    'days': 0,
                    'added': 0
                }
                continue
            
            # Save the data
            candles = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].values.tolist()
            candles = [[int(row[0].timestamp() * 1000)] + list(row[1:]) for row in candles]
            
            saved = self.db_ops.save_ohlcv_data(symbol, '1h', candles)
            
            # Get final stats
            final_count = self.db_ops.get_candle_count(symbol)
            final_oldest = self.db_ops.get_oldest_candle(symbol)
            
            if final_oldest:
                final_days = (datetime.now(timezone.utc) - final_oldest['timestamp']).days
                date_range = f"{final_oldest['timestamp'].date()} to now"
                
                logger.info(f"   ✅ Success!")
                logger.info(f"   Added: {saved} new candles")
                logger.info(f"   Total: {final_count} candles ({final_days} days)")
                logger.info(f"   Range: {date_range}")
                
                results[symbol] = {
                    'status': 'success',
                    'candles': final_count,
                    'days': final_days,
                    'added': saved,
                    'date_range': date_range
                }
            
        # Summary
        logger.info("\n" + "="*50)
        logger.info("📊 BACKFILL SUMMARY")
        logger.info("="*50)
        for symbol, result in results.items():
            if result['status'] == 'success':
                logger.info(f"✅ {symbol}: {result['days']} days, {result['added']} new candles")
            elif result['status'] == 'sufficient':
                logger.info(f"✅ {symbol}: Already has {result['days']} days")
            else:
                logger.info(f"❌ {symbol}: Failed to fetch data")
        
        return results
        
    def validate_data_quality(self, symbols: List[str] = None) -> Dict[str, Dict]:
        """Validate data quality for all symbols.
        
        Args:
            symbols: List of symbols to validate
            
        Returns:
            Dictionary with quality metrics
        """
        if symbols is None:
            symbols = config.assets
            
        quality_report = {}
        
        for symbol in symbols:
            df = self.db_ops.get_latest_candles(symbol, '1h', limit=1000)
            
            if df.empty:
                quality_report[symbol] = {'status': 'NO_DATA', 'issues': ['No data available']}
                continue
                
            issues = []
            
            # Check for missing values
            missing_values = df.isnull().sum().sum()
            if missing_values > 0:
                issues.append(f"Missing values: {missing_values}")
                
            # Check for duplicate timestamps
            duplicate_timestamps = df.duplicated('timestamp').sum()
            if duplicate_timestamps > 0:
                issues.append(f"Duplicate timestamps: {duplicate_timestamps}")
                
            # Check for zero volume
            zero_volume = (df['volume'] == 0).sum()
            if zero_volume > 0:
                issues.append(f"Zero volume candles: {zero_volume}")
                
            # Check for price anomalies
            price_spread = df['high'] - df['low']
            extreme_spreads = (price_spread > price_spread.quantile(0.99)).sum()
            if extreme_spreads > 10:  # More than 1% extreme spreads
                issues.append(f"Extreme price spreads: {extreme_spreads}")
                
            quality_report[symbol] = {
                'status': 'VALID' if len(issues) == 0 else 'ISSUES_FOUND',
                'total_candles': len(df),
                'issues': issues,
                'quality_score': max(0, 100 - len(issues) * 20)
            }
            
        return quality_report

    def _initialize_futures_fetcher(self):
        """Lazy initialization of futures fetcher."""
        if self.futures_fetcher is None:
            from src.data.futures_fetcher import FuturesDataFetcher
            self.futures_fetcher = FuturesDataFetcher()
            logger.info("Initialized futures data fetcher")
    
    def ensure_futures_data_availability(self, symbol: str, min_hours: int = 2000) -> bool:
        """Ensure minimum futures data is available.
        
        Args:
            symbol: Trading pair symbol
            min_hours: Minimum hours of data required
            
        Returns:
            True if data coverage is sufficient
        """
        self._initialize_futures_fetcher()
        
        # First ensure the futures table exists
        try:
            self.db_ops.create_futures_table()
        except Exception as e:
            logger.error(f"Error creating futures table: {e}")
            return False
        
        # Check existing futures data
        futures_df = self.db_ops.get_latest_futures_data(symbol, hours_back=min_hours)
        
        if futures_df.empty or len(futures_df) < min_hours:
            logger.info(f"📊 Fetching futures data for {symbol} (have {len(futures_df)} hours, need {min_hours})")
            
            # Fetch historical OI data
            try:
                # Fetch open interest history (now supports extended history)
                days_to_fetch = (min_hours // 24) + 1
                logger.info(f"   Fetching up to {days_to_fetch} days of futures data...")
                oi_df = self.futures_fetcher.fetch_historical_open_interest(
                    symbol, period='1h', days_back=days_to_fetch
                )
                
                if not oi_df.empty:
                    logger.info(f"   Fetched {len(oi_df)} hours of OI data")
                    
                    # Fetch funding rates separately (they update every 8 hours, not hourly)
                    funding_df = self.futures_fetcher.fetch_funding_history(
                        symbol, days_back=min(days_to_fetch, 90)  # Funding data limited to ~90 days
                    )
                    
                    if not funding_df.empty:
                        logger.info(f"   Fetched {len(funding_df)} funding rate records (8-hour intervals)")
                    
                    # Save futures data
                    saved_count = 0
                    for _, row in oi_df.iterrows():
                        timestamp = row['timestamp']
                        
                        # Find nearest funding rate
                        funding_rate = None
                        if not funding_df.empty:
                            time_diffs = abs((funding_df['timestamp'] - timestamp).dt.total_seconds())
                            if time_diffs.min() < 3600:  # Within 1 hour
                                nearest_idx = time_diffs.idxmin()
                                funding_rate = funding_df.loc[nearest_idx, 'funding_rate']
                        
                        # Save to database
                        success = self.db_ops.save_futures_data(
                            symbol=symbol,
                            timestamp=timestamp,
                            open_interest=row.get('open_interest'),
                            open_interest_value=row.get('open_interest_value'),
                            funding_rate=funding_rate
                        )
                        
                        if success:
                            saved_count += 1
                    
                    logger.info(f"   ✅ Saved {saved_count} futures data points for {symbol}")
                    
                else:
                    logger.warning(f"   ⚠️ No historical futures data available for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error fetching futures data for {symbol}: {e}")
                return False
                
        return True
    
    def ensure_all_symbols_futures_coverage(self, min_hours: int = None, target_years: int = 1) -> Dict[str, bool]:
        """Ensure all symbols have maximum possible futures data coverage.
        
        Args:
            min_hours: Minimum hours of futures data required. If None, tries to match spot data or target_years.
            target_years: Target years of historical futures data to fetch (default 1 year)
            
        Returns:
            Dictionary mapping symbols to success status
        """
        self._initialize_futures_fetcher()
        
        results = {}
        
        print("\n📊 Futures Data Status Check:")
        print("   " + "-" * 45)
        
        for symbol in config.assets:
            try:
                # Determine target coverage
                if min_hours is None:
                    # Try to match spot data coverage, but cap at target_years
                    spot_count = self.db_ops.get_candle_count(symbol)
                    oldest_spot = self.db_ops.get_oldest_candle(symbol)
                    
                    if oldest_spot:
                        spot_days = (datetime.now(timezone.utc) - oldest_spot['timestamp']).days
                        # Target at least 1 year of futures data, or match spot if less
                        target_days = min(spot_days, target_years * 365)
                    else:
                        target_days = target_years * 365
                    
                    target_hours = target_days * 24
                else:
                    target_hours = min_hours
                    target_days = target_hours // 24
                
                # Check current futures data
                futures_df = self.db_ops.get_latest_futures_data(symbol, hours_back=target_hours)
                current_hours = len(futures_df)
                
                if current_hours < target_hours:
                    print(f"   ⚠️  {symbol}: {current_hours}/{target_hours} hours of futures data")
                    print(f"      → Will attempt to fetch {target_days} days of futures data...")
                    
                    # Fetch extended futures data
                    success = self.ensure_futures_data_availability(symbol, target_hours)
                    results[symbol] = success
                    
                    if success:
                        # Check how much we actually got
                        new_futures_df = self.db_ops.get_latest_futures_data(symbol, hours_back=target_hours)
                        new_hours = len(new_futures_df)
                        print(f"      ✅ Futures data updated: {new_hours} hours available")
                else:
                    print(f"   ✅ {symbol}: {current_hours} hours of futures data available")
                    results[symbol] = True
                
            except Exception as e:
                logger.error(f"Error checking futures data for {symbol}: {e}")
                results[symbol] = False
                
        return results
    
    def fetch_latest_futures_data(self, symbols: List[str] = None) -> Dict[str, Dict]:
        """Fetch latest futures data for all symbols.
        
        Args:
            symbols: List of symbols to fetch
            
        Returns:
            Dictionary with fetched data
        """
        self._initialize_futures_fetcher()
        
        if symbols is None:
            symbols = config.assets
            
        results = {}
        
        for symbol in symbols:
            try:
                # Fetch all futures data
                futures_data = self.futures_fetcher.fetch_all_futures_data(symbol)
                
                if futures_data and 'timestamp' in futures_data:
                    # Save to database
                    success = self.db_ops.save_futures_data(
                        symbol=symbol,
                        timestamp=futures_data['timestamp'],
                        open_interest=futures_data.get('open_interest'),
                        funding_rate=futures_data.get('funding_rate'),
                        top_trader_ratio=futures_data.get('top_trader_ratio'),
                        taker_buy_sell_ratio=futures_data.get('taker_buy_sell_ratio')
                    )
                    
                    if success:
                        logger.info(f"💾 Saved futures data for {symbol}: OI={futures_data.get('open_interest', 'N/A')}, FR={futures_data.get('funding_rate', 'N/A')}")
                        results[symbol] = futures_data
                    else:
                        logger.error(f"Failed to save futures data for {symbol}")
                        results[symbol] = None
                else:
                    logger.warning(f"No futures data fetched for {symbol}")
                    results[symbol] = None
                    
            except Exception as e:
                logger.error(f"Error fetching futures data for {symbol}: {e}")
                results[symbol] = None
                
        return results


# Create singleton instance
recovery_manager = DataRecoveryManager()
