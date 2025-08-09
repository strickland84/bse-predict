"""Futures data fetcher for open interest, funding rates, and other futures metrics."""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import ccxt
import pandas as pd
import requests


class FuturesDataFetcher:
    """Fetches futures market data including open interest and funding rates."""
    
    def __init__(self):
        """Initialize Binance futures exchange connection."""
        self.exchange = ccxt.binance({
            'apiKey': '',  # Not needed for public data
            'secret': '',
            'options': {
                'defaultType': 'future',  # Use futures endpoints
            }
        })
        self.logger = logging.getLogger(__name__)
        
    def fetch_open_interest(self, symbol: str) -> Optional[Dict]:
        """Fetch current open interest for a symbol.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            
        Returns:
            Dictionary with open interest data or None if error
        """
        try:
            # Convert symbol format: BTC/USDT -> BTCUSDT
            futures_symbol = symbol.replace('/', '')
            
            # Use direct REST API call for current open interest
            url = 'https://fapi.binance.com/fapi/v1/openInterest'
            params = {'symbol': futures_symbol}
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            return {
                'symbol': symbol,
                'open_interest': float(data['openInterest']),
                'timestamp': pd.to_datetime(data['time'], unit='ms')
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching OI for {symbol}: {e}")
            return None
            
    def fetch_historical_open_interest(self, symbol: str, 
                                     period: str = '1h',
                                     days_back: int = 30) -> pd.DataFrame:
        """Fetch historical open interest data with support for extended history.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            period: Time period (5m, 15m, 30m, 1h, 4h, 1d)
            days_back: Number of days to fetch (supports up to years of data)
            
        Returns:
            DataFrame with historical OI data
        """
        try:
            futures_symbol = symbol.replace('/', '')
            end_time = int(datetime.now().timestamp() * 1000)
            start_time = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
            
            all_data = []
            current_end = end_time
            
            # Fetch in chunks of 500 records (API limit)
            while current_end > start_time:
                # Binance futures data endpoint
                url = 'https://fapi.binance.com/futures/data/openInterestHist'
                params = {
                    'symbol': futures_symbol,
                    'period': period,
                    'endTime': current_end,
                    'limit': 500  # Max limit per request
                }
                
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if not data:
                    break
                
                all_data.extend(data)
                
                # Move to next chunk
                oldest_timestamp = min(item['timestamp'] for item in data)
                if oldest_timestamp <= start_time or len(data) < 500:
                    break
                    
                current_end = oldest_timestamp - 1
                
                # Rate limiting for large requests
                import time
                time.sleep(0.1)
            
            if not all_data:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(all_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['open_interest'] = df['sumOpenInterest'].astype(float)
            df['open_interest_value'] = df['sumOpenInterestValue'].astype(float)
            
            # Remove duplicates and sort
            df = df.drop_duplicates('timestamp').sort_values('timestamp')
            
            # Filter to requested time range
            df = df[df['timestamp'] >= pd.to_datetime(start_time, unit='ms')]
            
            self.logger.info(f"Fetched {len(df)} OI records for {symbol} ({days_back} days requested)")
            
            return df[['timestamp', 'open_interest', 'open_interest_value']]
            
        except Exception as e:
            self.logger.error(f"Error fetching historical OI for {symbol}: {e}")
            return pd.DataFrame()
            
    def fetch_funding_rate(self, symbol: str) -> Optional[float]:
        """Fetch current funding rate.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            
        Returns:
            Current funding rate or None if error
        """
        try:
            futures_symbol = symbol.replace('/', '')
            
            # Use direct REST API call for funding rate
            url = 'https://fapi.binance.com/fapi/v1/fundingRate'
            params = {
                'symbol': futures_symbol,
                'limit': 1
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data and len(data) > 0:
                return float(data[0]['fundingRate'])
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching funding rate for {symbol}: {e}")
            return None
            
    def fetch_funding_history(self, symbol: str, days_back: int = 7) -> pd.DataFrame:
        """Fetch historical funding rates with support for extended history.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            days_back: Number of days to fetch (supports years of data)
            
        Returns:
            DataFrame with funding rate history
        """
        try:
            futures_symbol = symbol.replace('/', '')
            end_time = int(datetime.now().timestamp() * 1000)
            start_time = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
            
            all_data = []
            current_end = end_time
            
            # Fetch in chunks of 1000 records (API limit)
            while current_end > start_time:
                # Use direct REST API call for funding rate history
                url = 'https://fapi.binance.com/fapi/v1/fundingRate'
                params = {
                    'symbol': futures_symbol,
                    'endTime': current_end,
                    'limit': 1000  # Max limit per request
                }
                
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if not data:
                    break
                
                all_data.extend(data)
                
                # Move to next chunk
                oldest_timestamp = min(item['fundingTime'] for item in data)
                if oldest_timestamp <= start_time or len(data) < 1000:
                    break
                    
                current_end = oldest_timestamp - 1
                
                # Rate limiting for large requests
                import time
                time.sleep(0.1)
            
            if not all_data:
                return pd.DataFrame()
                
            df = pd.DataFrame(all_data)
            df['timestamp'] = pd.to_datetime(df['fundingTime'], unit='ms')
            df['funding_rate'] = df['fundingRate'].astype(float)
            
            # Remove duplicates and sort
            df = df.drop_duplicates('timestamp').sort_values('timestamp')
            
            # Filter to requested time range
            df = df[df['timestamp'] >= pd.to_datetime(start_time, unit='ms')]
            
            self.logger.info(f"Fetched {len(df)} funding rate records for {symbol} ({days_back} days requested)")
            
            return df[['timestamp', 'funding_rate']]
            
        except Exception as e:
            self.logger.error(f"Error fetching funding history for {symbol}: {e}")
            return pd.DataFrame()
            
    def fetch_top_trader_ratio(self, symbol: str, period: str = '5m') -> Optional[Dict]:
        """Fetch top trader long/short account ratio.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            period: Time period (5m, 15m, 30m, 1h, 4h, 1d)
            
        Returns:
            Dictionary with ratio data or None if error
        """
        try:
            futures_symbol = symbol.replace('/', '')
            
            url = 'https://fapi.binance.com/futures/data/topLongShortAccountRatio'
            params = {
                'symbol': futures_symbol,
                'period': period,
                'limit': 1
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data and len(data) > 0:
                return {
                    'long_short_ratio': float(data[0]['longShortRatio']),
                    'long_account': float(data[0]['longAccount']),
                    'short_account': float(data[0]['shortAccount']),
                    'timestamp': pd.to_datetime(data[0]['timestamp'], unit='ms')
                }
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching top trader ratio for {symbol}: {e}")
            return None
            
    def fetch_taker_buy_sell_volume(self, symbol: str, period: str = '5m') -> Optional[Dict]:
        """Fetch taker buy/sell volume ratio.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            period: Time period (5m, 15m, 30m, 1h, 4h, 1d)
            
        Returns:
            Dictionary with volume data or None if error
        """
        try:
            futures_symbol = symbol.replace('/', '')
            
            url = 'https://fapi.binance.com/futures/data/takerlongshortRatio'
            params = {
                'symbol': futures_symbol,
                'period': period,
                'limit': 1
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data and len(data) > 0:
                return {
                    'buy_sell_ratio': float(data[0]['buySellRatio']),
                    'buy_vol': float(data[0]['buyVol']),
                    'sell_vol': float(data[0]['sellVol']),
                    'timestamp': pd.to_datetime(data[0]['timestamp'], unit='ms')
                }
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching taker volume for {symbol}: {e}")
            return None
            
    def fetch_all_futures_data(self, symbol: str) -> Dict:
        """Fetch all available futures data for a symbol.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            
        Returns:
            Dictionary with all futures data
        """
        data = {
            'symbol': symbol,
            'timestamp': datetime.now()
        }
        
        # Fetch all data types
        oi_data = self.fetch_open_interest(symbol)
        if oi_data:
            data['open_interest'] = oi_data['open_interest']
            
        funding_rate = self.fetch_funding_rate(symbol)
        if funding_rate is not None:
            data['funding_rate'] = funding_rate
            
        top_trader = self.fetch_top_trader_ratio(symbol)
        if top_trader:
            data['top_trader_ratio'] = top_trader['long_short_ratio']
            
        taker_volume = self.fetch_taker_buy_sell_volume(symbol)
        if taker_volume:
            data['taker_buy_sell_ratio'] = taker_volume['buy_sell_ratio']
            
        return data


# Singleton instance
futures_fetcher = FuturesDataFetcher()