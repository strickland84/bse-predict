"""Feature engineering for multi-target cryptocurrency price prediction."""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MultiTargetFeatureEngineer:
    """Creates comprehensive features for multi-target price prediction."""
    
    def __init__(self, target_percentages: List[float] = None):
        """Initialize feature engineer.
        
        Args:
            target_percentages: List of target percentages [0.01, 0.02, 0.05]
        """
        self.target_percentages = target_percentages or [0.01, 0.02, 0.05]
        self.feature_version = 'v1.0'
        
    def create_features(self, df: pd.DataFrame) -> Optional[Dict]:
        """Create comprehensive feature set from OHLCV data.
        
        Args:
            df: DataFrame with columns: timestamp, open, high, low, close, volume
            
        Returns:
            Dictionary of features or None if insufficient data
        """
        if df.empty or len(df) < 50:
            logger.warning("Insufficient data for feature creation")
            return None
            
        try:
            features = {}
            df = df.copy()
            
            # Ensure timestamp is datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Basic price features
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df['high_low_pct'] = (df['high'] - df['low']) / df['close']
            df['body_pct'] = abs(df['close'] - df['open']) / df['close']
            df['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
            df['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
            
            # Multi-timeframe indicators
            periods = [6, 12, 24, 48, 168]  # 6h, 12h, 1d, 2d, 1w
            
            for period in periods:
                if len(df) >= period:
                    # Moving averages
                    sma = df['close'].rolling(period, min_periods=period//2).mean()
                    ema = df['close'].ewm(span=period, adjust=False).mean()
                    
                    if not sma.empty and sma.iloc[-1] > 0:
                        features[f'price_vs_sma_{period}'] = df['close'].iloc[-1] / sma.iloc[-1] - 1
                        features[f'price_vs_ema_{period}'] = df['close'].iloc[-1] / ema.iloc[-1] - 1
                        features[f'sma_trend_{period}'] = (sma.iloc[-1] / sma.iloc[-period//2]) - 1
                    
                    # Bollinger Bands
                    bb_std = df['close'].rolling(period, min_periods=period//2).std()
                    bb_upper = sma + (2 * bb_std)
                    bb_lower = sma - (2 * bb_std)
                    
                    if not bb_std.empty and bb_std.iloc[-1] > 0:
                        bb_width = bb_upper.iloc[-1] - bb_lower.iloc[-1]
                        features[f'bb_position_{period}'] = (df['close'].iloc[-1] - bb_lower.iloc[-1]) / bb_width
                        features[f'bb_width_{period}'] = bb_width / df['close'].iloc[-1]
                    
                    # RSI
                    delta = df['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(period, min_periods=period//2).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(period, min_periods=period//2).mean()
                    
                    if not loss.empty and loss.iloc[-1] > 0:
                        rs = gain.iloc[-1] / loss.iloc[-1]
                        rsi = 100 - (100 / (1 + rs))
                        features[f'rsi_{period}'] = rsi
                    
                    # Volume indicators
                    volume_sma = df['volume'].rolling(period, min_periods=period//2).mean()
                    if volume_sma.iloc[-1] > 0:
                        features[f'volume_ratio_{period}'] = df['volume'].iloc[-1] / volume_sma.iloc[-1]
                        features[f'volume_trend_{period}'] = (volume_sma.iloc[-1] / volume_sma.iloc[-period//2]) - 1
            
            # Volatility measures
            returns = df['returns'].dropna()
            if len(returns) >= 24:
                features['volatility_24h'] = returns.tail(24).std()
                features['volatility_168h'] = returns.tail(168).std()
                
                if features['volatility_168h'] > 0:
                    features['volatility_regime'] = features['volatility_24h'] / features['volatility_168h']
            
            # Momentum indicators
            for period in [6, 12, 24]:
                if len(returns) >= period:
                    recent_returns = returns.tail(period)
                    features[f'momentum_{period}'] = recent_returns.mean()
                    features[f'momentum_consistency_{period}'] = (recent_returns > 0).mean()
                    
                    # Rate of change
                    if len(df) >= period:
                        roc = (df['close'].iloc[-1] / df['close'].iloc[-period]) - 1
                        features[f'roc_{period}'] = roc
            
            # Price position in range
            for period in [24, 48, 168]:
                if len(df) >= period:
                    high_max = df['high'].tail(period).max()
                    low_min = df['low'].tail(period).min()
                    price_range = high_max - low_min
                    
                    if price_range > 0:
                        features[f'price_position_{period}'] = (df['close'].iloc[-1] - low_min) / price_range
                        features[f'distance_to_high_{period}'] = (high_max - df['close'].iloc[-1]) / df['close'].iloc[-1]
                        features[f'distance_to_low_{period}'] = (df['close'].iloc[-1] - low_min) / df['close'].iloc[-1]
            
            # Lag features
            for lag in [1, 2, 3, 6, 12, 24]:
                if len(returns) > lag:
                    features[f'return_lag_{lag}'] = returns.iloc[-lag-1] if not pd.isna(returns.iloc[-lag-1]) else 0
                    
                    if lag <= 6:
                        vol_24 = returns.tail(24).std()
                        features[f'volatility_lag_{lag}'] = vol_24
            
            # Time-based features
            if 'timestamp' in df.columns and len(df) > 0:
                latest_timestamp = df['timestamp'].iloc[-1]
                if isinstance(latest_timestamp, str):
                    latest_timestamp = pd.to_datetime(latest_timestamp)
                
                hour = latest_timestamp.hour
                features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
                features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
                features['hour_of_day'] = hour
                features['is_weekend'] = 1 if latest_timestamp.weekday() >= 5 else 0
                features['day_of_week'] = latest_timestamp.weekday()
                
                # Market session features
                features['is_asian_session'] = 1 if 0 <= hour < 8 else 0
                features['is_european_session'] = 1 if 8 <= hour < 16 else 0
                features['is_american_session'] = 1 if 16 <= hour < 24 else 0
            
            # Market microstructure features
            if len(df) >= 24:
                # Average true range
                high_low = df['high'] - df['low']
                high_close = abs(df['high'] - df['close'].shift(1))
                low_close = abs(df['low'] - df['close'].shift(1))
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr = true_range.rolling(14, min_periods=7).mean()
                
                if not atr.empty and atr.iloc[-1] > 0:
                    features['atr_ratio'] = atr.iloc[-1] / df['close'].iloc[-1]
            
            # Futures data features (if available)
            if 'open_interest' in df.columns:
                # Open Interest features
                oi = df['open_interest'].ffill()
                
                if len(oi) >= 2 and not oi.empty:
                    # OI change rates
                    if oi.iloc[-2] > 0:
                        features['oi_change_1h'] = (oi.iloc[-1] / oi.iloc[-2] - 1) if pd.notna(oi.iloc[-1]) and pd.notna(oi.iloc[-2]) else 0
                    
                    if len(oi) >= 24 and oi.iloc[-24] > 0:
                        features['oi_change_24h'] = (oi.iloc[-1] / oi.iloc[-24] - 1) if pd.notna(oi.iloc[-1]) and pd.notna(oi.iloc[-24]) else 0
                    
                    # OI moving averages
                    if len(oi) >= 24:
                        oi_ma_24 = oi.rolling(24, min_periods=12).mean().iloc[-1]
                        if pd.notna(oi_ma_24) and oi_ma_24 > 0:
                            features['oi_ma_ratio'] = oi.iloc[-1] / oi_ma_24 if pd.notna(oi.iloc[-1]) else 1
                    
                    # OI velocity (rate of change acceleration)
                    if len(oi) >= 3:
                        oi_changes = oi.pct_change(fill_method=None).fillna(0)
                        if pd.notna(oi_changes.iloc[-1]) and pd.notna(oi_changes.iloc[-2]):
                            features['oi_velocity'] = oi_changes.iloc[-1] - oi_changes.iloc[-2]
                
                # Price vs OI divergence
                if len(df) >= 24:
                    price_changes = df['close'].pct_change(fill_method=None).fillna(0)
                    oi_changes = oi.pct_change(fill_method=None).fillna(0)
                    
                    # Rolling correlation
                    if len(price_changes) >= 24 and len(oi_changes) >= 24:
                        # Only calculate correlation if we have variance in both series
                        if price_changes.tail(24).std() > 0 and oi_changes.tail(24).std() > 0:
                            price_oi_corr = price_changes.tail(24).corr(oi_changes.tail(24))
                            features['price_oi_corr_24h'] = price_oi_corr if pd.notna(price_oi_corr) else 0
                        else:
                            features['price_oi_corr_24h'] = 0
                    
                    # Divergence detection
                    if pd.notna(df['close'].iloc[-1]) and pd.notna(df['close'].iloc[-24]) and df['close'].iloc[-24] > 0:
                        recent_price_trend = (df['close'].iloc[-1] / df['close'].iloc[-24] - 1)
                        if pd.notna(oi.iloc[-1]) and pd.notna(oi.iloc[-24]) and oi.iloc[-24] > 0:
                            recent_oi_trend = (oi.iloc[-1] / oi.iloc[-24] - 1)
                            
                            # Bullish divergence: Price down but OI up
                            if recent_price_trend < -0.01 and recent_oi_trend > 0.01:
                                features['oi_bullish_divergence'] = 1
                            else:
                                features['oi_bullish_divergence'] = 0
                                
                            # Bearish divergence: Price up but OI down  
                            if recent_price_trend > 0.01 and recent_oi_trend < -0.01:
                                features['oi_bearish_divergence'] = 1
                            else:
                                features['oi_bearish_divergence'] = 0
                
                # Volume to OI ratio
                if pd.notna(df['volume'].iloc[-1]) and pd.notna(oi.iloc[-1]) and oi.iloc[-1] > 0:
                    features['volume_oi_ratio'] = df['volume'].iloc[-1] / oi.iloc[-1]
            
            # Funding rate features (if available)
            if 'funding_rate' in df.columns:
                fr = df['funding_rate'].ffill()
                
                if not fr.empty and pd.notna(fr.iloc[-1]):
                    features['funding_rate'] = fr.iloc[-1]
                    features['funding_rate_extreme'] = 1 if abs(fr.iloc[-1]) > 0.0005 else 0
                    
                    # Funding rate momentum
                    if len(fr) >= 8:
                        fr_ma = fr.rolling(8, min_periods=4).mean().iloc[-1]
                        if pd.notna(fr_ma):
                            features['funding_rate_momentum'] = fr.iloc[-1] - fr_ma
                    
                    # Cumulative funding over periods
                    if len(fr) >= 24:
                        features['funding_cumulative_24h'] = fr.tail(24).sum()
                    if len(fr) >= 72:
                        features['funding_cumulative_72h'] = fr.tail(72).sum()
            
            # Top trader ratio features (if available)
            if 'top_trader_ratio' in df.columns:
                ttr = df['top_trader_ratio'].ffill()
                
                if not ttr.empty and pd.notna(ttr.iloc[-1]):
                    features['top_trader_ratio'] = ttr.iloc[-1]
                    features['top_trader_bullish'] = 1 if ttr.iloc[-1] > 1.2 else 0
                    features['top_trader_bearish'] = 1 if ttr.iloc[-1] < 0.8 else 0
            
            # Taker buy/sell ratio features (if available)
            if 'taker_buy_sell_ratio' in df.columns:
                tbsr = df['taker_buy_sell_ratio'].ffill()
                
                if not tbsr.empty and pd.notna(tbsr.iloc[-1]):
                    features['taker_buy_sell_ratio'] = tbsr.iloc[-1]
                    features['taker_buy_pressure'] = 1 if tbsr.iloc[-1] > 1.1 else 0
                    features['taker_sell_pressure'] = 1 if tbsr.iloc[-1] < 0.9 else 0
            
            # Clean up any NaN values
            features = {k: (0.0 if pd.isna(v) else float(v)) for k, v in features.items()}
            
            # Don't add non-numeric metadata to features used for ML
            # features['feature_version'] = self.feature_version
            # features['feature_count'] = len(features)
            
            # logger.info(f"Created {len(features)} features for latest timestamp")  # Too verbose
            return features
            
        except Exception as e:
            logger.error(f"Error creating features: {e}")
            return None
    
    def create_targets_for_all_percentages(self, df: pd.DataFrame, start_idx: int, 
                                         max_hours: int = 72) -> Dict[float, Optional[int]]:
        """Create binary targets for all percentage levels.
        
        Args:
            df: DataFrame with OHLCV data
            start_idx: Index to start prediction from
            max_hours: Maximum hours to look ahead
            
        Returns:
            Dictionary mapping target percentages to binary targets (0=DOWN, 1=UP)
        """
        targets = {}
        
        if start_idx >= len(df) - 2:  # Need at least 2 future candles
            return {pct: None for pct in self.target_percentages}
            
        start_price = df['close'].iloc[start_idx]
        
        for target_pct in self.target_percentages:
            target_up = start_price * (1 + target_pct)
            target_down = start_price * (1 - target_pct)
            
            # Look forward to see which target gets hit first
            target_hit = None
            
            for j in range(start_idx + 1, min(start_idx + max_hours + 1, len(df))):
                high = df['high'].iloc[j]
                low = df['low'].iloc[j]
                
                # Check BOTH conditions - whichever hits first wins
                # Note: If both hit in same candle, we can't determine order, so we check both
                up_hit = high >= target_up
                down_hit = low <= target_down
                
                if up_hit and down_hit:
                    # Both targets hit in same candle - this is ambiguous
                    # We could skip these, but they're rare. For now, prefer the larger move
                    up_move = (high / start_price) - 1
                    down_move = 1 - (low / start_price)
                    if up_move >= down_move:
                        target_hit = 1  # UP
                    else:
                        target_hit = 0  # DOWN
                    break
                elif up_hit:
                    target_hit = 1  # UP
                    break
                elif down_hit:
                    target_hit = 0  # DOWN
                    break
            
            targets[target_pct] = target_hit  # Will be None if no target hit
                
        return targets
    
    def prepare_training_datasets(self, df: pd.DataFrame) -> Dict[float, Tuple[pd.DataFrame, pd.Series]]:
        """Prepare separate datasets for each target percentage.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary mapping target percentages to (X, y) tuples
        """
        if len(df) < 200:  # Need sufficient data
            logger.warning(f"Insufficient data: {len(df)} candles, need 200+")
            return {}
            
        datasets = {pct: (None, None) for pct in self.target_percentages}
        
        # Collect features and targets
        feature_list = []
        all_targets = {pct: [] for pct in self.target_percentages}
        
        # Process data (leave buffer for future target calculation)
        start_idx = 168  # 1 week of lookback needed
        end_idx = len(df) - 72  # 72 hours buffer for target calculation
        
        logger.info(f"Processing {end_idx - start_idx} samples for feature engineering...")
        
        for i in range(start_idx, end_idx):
            if i % 100 == 0 and i > start_idx:
                logger.info(f"Processing sample {i - start_idx + 1}/{end_idx - start_idx}")
                
            # Get window for feature calculation
            window_df = df.iloc[max(0, i-168):i+1].copy()
            
            if len(window_df) < 50:
                continue
                
            # Create features
            features = self.create_features(window_df)
            if features is None:
                continue
                
            # Create targets for all percentages
            targets = self.create_targets_for_all_percentages(df, i)
            
            # Only keep samples where we have at least one valid target
            if any(target is not None for target in targets.values()):
                feature_list.append(features)
                for pct in self.target_percentages:
                    all_targets[pct].append(targets[pct])
        
        if not feature_list:
            logger.warning("No valid samples found")
            return datasets
            
        # Convert to DataFrames for each target
        X_all = pd.DataFrame(feature_list)
        
        for target_pct in self.target_percentages:
            y = pd.Series(all_targets[target_pct])
            
            # Remove samples with missing targets for this percentage
            valid_mask = ~y.isnull()
            X_clean = X_all[valid_mask].copy()
            y_clean = y[valid_mask].copy()
            
            # Remove samples with missing features
            feature_mask = ~X_clean.isnull().any(axis=1)
            X_final = X_clean[feature_mask]
            y_final = y_clean[feature_mask]
            
            if len(X_final) >= 100:  # Minimum samples needed
                datasets[target_pct] = (X_final, y_final)
                logger.info(f"Target {target_pct:.1%}: {len(X_final)} samples, "
                          f"distribution: {y_final.value_counts(normalize=True).to_dict()}")
            else:
                logger.warning(f"Target {target_pct:.1%}: Insufficient samples ({len(X_final)})")
                
        return datasets
    
    def prepare_training_datasets_with_timestamps(self, df: pd.DataFrame) -> Dict[float, Tuple[pd.DataFrame, pd.Series, pd.Series]]:
        """Prepare datasets for each target percentage and include sample timestamps.
        
        Args:
            df: DataFrame with OHLCV (and optional futures) data
        
        Returns:
            Dictionary mapping target percentages to (X, y, timestamps) tuples.
            If insufficient samples for a target, the tuple will be (None, None, None).
        """
        if len(df) < 200:
            logger.warning(f"Insufficient data: {len(df)} candles, need 200+")
            return {pct: (None, None, None) for pct in self.target_percentages}
        
        datasets_ts = {pct: (None, None, None) for pct in self.target_percentages}
        
        feature_list: List[Dict] = []
        all_targets = {pct: [] for pct in self.target_percentages}
        timestamps_list: List[pd.Timestamp] = []
        
        start_idx = 168  # 1 week of lookback needed
        end_idx = len(df) - 72  # 72 hours buffer for target calculation
        
        logger.info(f"Processing {end_idx - start_idx} samples for feature engineering (with timestamps)...")
        
        for i in range(start_idx, end_idx):
            if i % 100 == 0 and i > start_idx:
                logger.info(f"Processing sample {i - start_idx + 1}/{end_idx - start_idx}")
            
            window_df = df.iloc[max(0, i-168):i+1].copy()
            if len(window_df) < 50:
                continue
            
            features = self.create_features(window_df)
            if features is None:
                continue
            
            targets = self.create_targets_for_all_percentages(df, i)
            
            if any(target is not None for target in targets.values()):
                feature_list.append(features)
                timestamps_list.append(df['timestamp'].iloc[i] if 'timestamp' in df.columns else pd.NaT)
                for pct in self.target_percentages:
                    all_targets[pct].append(targets[pct])
        
        if not feature_list:
            logger.warning("No valid samples found")
            return datasets_ts
        
        X_all = pd.DataFrame(feature_list)
        ts_all = pd.to_datetime(pd.Series(timestamps_list))
        
        for target_pct in self.target_percentages:
            y = pd.Series(all_targets[target_pct])
            valid_mask = ~y.isnull()
            X_clean = X_all[valid_mask].copy()
            y_clean = y[valid_mask].copy()
            ts_clean = ts_all[valid_mask].copy()
            
            feature_mask = ~X_clean.isnull().any(axis=1)
            X_final = X_clean[feature_mask]
            y_final = y_clean[feature_mask]
            ts_final = ts_clean[feature_mask]
            
            if len(X_final) >= 100:
                datasets_ts[target_pct] = (X_final, y_final, ts_final)
                logger.info(f"Target {target_pct:.1%}: {len(X_final)} samples (with timestamps), "
                            f"distribution: {y_final.value_counts(normalize=True).to_dict()}")
            else:
                logger.warning(f"Target {target_pct:.1%}: Insufficient samples ({len(X_final)})")
        
        return datasets_ts
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names."""
        # This is a static list based on the features created
        base_features = [
            'returns', 'log_returns', 'high_low_pct', 'body_pct', 'upper_wick', 'lower_wick'
        ]
        
        periods = [6, 12, 24, 48, 168]
        indicator_features = []
        
        for period in periods:
            indicator_features.extend([
                f'price_vs_sma_{period}', f'price_vs_ema_{period}', f'sma_trend_{period}',
                f'bb_position_{period}', f'bb_width_{period}', f'rsi_{period}',
                f'volume_ratio_{period}', f'volume_trend_{period}'
            ])
        
        volatility_features = ['volatility_24h', 'volatility_168h', 'volatility_regime']
        
        momentum_features = []
        for period in [6, 12, 24]:
            momentum_features.extend([
                f'momentum_{period}', f'momentum_consistency_{period}', f'roc_{period}'
            ])
        
        position_features = []
        for period in [24, 48, 168]:
            position_features.extend([
                f'price_position_{period}', f'distance_to_high_{period}', f'distance_to_low_{period}'
            ])
        
        lag_features = [f'return_lag_{lag}' for lag in [1, 2, 3, 6, 12, 24]]
        vol_lag_features = [f'volatility_lag_{lag}' for lag in [1, 2, 3, 6]]
        
        time_features = ['hour_sin', 'hour_cos', 'hour_of_day', 'is_weekend', 'day_of_week']
        session_features = ['is_asian_session', 'is_european_session', 'is_american_session']
        
        micro_features = ['atr_ratio']
        
        # Futures features
        futures_features = [
            'oi_change_1h', 'oi_change_24h', 'oi_ma_ratio', 'oi_velocity',
            'price_oi_corr_24h', 'oi_bullish_divergence', 'oi_bearish_divergence',
            'volume_oi_ratio', 'funding_rate', 'funding_rate_extreme',
            'funding_rate_momentum', 'funding_cumulative_24h', 'funding_cumulative_72h',
            'top_trader_ratio', 'top_trader_bullish', 'top_trader_bearish',
            'taker_buy_sell_ratio', 'taker_buy_pressure', 'taker_sell_pressure'
        ]
        
        all_features = (base_features + indicator_features + volatility_features + 
                       momentum_features + position_features + lag_features + 
                       vol_lag_features + time_features + session_features + 
                       micro_features + futures_features + ['feature_version', 'feature_count'])
        
        return all_features
