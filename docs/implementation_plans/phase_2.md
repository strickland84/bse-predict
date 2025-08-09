# PHASE 2: MACHINE LEARNING PIPELINE (Days 3-7)
**🎯 GOAL**: Build and validate multi-target prediction models

## What You're Building
- Feature engineering pipeline that converts raw price data into ML features
- Multi-target training system (9 models: 3 assets × 3 targets)
- Prediction engine that gives probability scores for each target
- Model evaluation and performance tracking system

---

## CHECKPOINT 2A: Feature Engineering (Day 3, 6 hours)

### Step 2.1: Multi-Target Feature Engineering
```python
# src/data/feature_engineer.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

class MultiTargetFeatureEngineer:
    def __init__(self, target_percentages: List[float] = [0.01, 0.02, 0.05]):
        self.target_percentages = target_percentages
        self.feature_version = 'v1'
        
    # NOTE: CCXT Limitations
    # CCXT primarily provides OHLCV data. For advanced features like open interest,
    # funding rates, or order book depth, you need to:
    # 1. Use exchange-specific APIs (Binance, Bybit have futures data)
    # 2. Consider alternative data providers (CryptoCompare, Glassnode)
    # 3. Build custom adapters for each exchange's websocket feeds
    # Current implementation focuses on technical indicators from OHLCV data
        
    def create_features(self, df: pd.DataFrame) -> Optional[Dict]:
        """Create comprehensive feature set"""
        if df.empty or len(df) < 50:
            return None
            
        features = {}
        
        # Basic price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['high_low_pct'] = (df['high'] - df['low']) / df['close']
        df['body_pct'] = abs(df['close'] - df['open']) / df['close']
        
        # Multi-timeframe indicators
        periods = [6, 12, 24, 48, 168]  # 6h, 12h, 1d, 2d, 1w
        
        for period in periods:
            if len(df) >= period:
                # Moving averages and price position
                sma = df['close'].rolling(period, min_periods=period//2).mean()
                if not sma.empty and sma.iloc[-1] > 0:
                    features[f'price_vs_sma_{period}'] = df['close'].iloc[-1] / sma.iloc[-1] - 1
                
                # Volatility (critical for different target sizes)
                vol = df['returns'].rolling(period, min_periods=period//2).std()
                if not vol.empty:
                    features[f'volatility_{period}'] = vol.iloc[-1]
                
                # Price position in range (important for breakouts)
                high_max = df['high'].rolling(period, min_periods=period//2).max()
                low_min = df['low'].rolling(period, min_periods=period//2).min()
                price_range = high_max.iloc[-1] - low_min.iloc[-1]
                if price_range > 0:
                    features[f'price_position_{period}'] = (df['close'].iloc[-1] - low_min.iloc[-1]) / price_range
                
                # Momentum indicators
                if period <= 48:  # Only for shorter periods
                    recent_returns = df['returns'].tail(period)
                    features[f'momentum_{period}'] = recent_returns.mean()
                    features[f'momentum_consistency_{period}'] = (recent_returns > 0).mean()
        
        # Recent pattern features (lag variables)
        for lag in [1, 2, 3, 6, 12, 24]:
            if len(df) > lag:
                features[f'return_lag_{lag}'] = df['returns'].iloc[-lag-1] if not pd.isna(df['returns'].iloc[-lag-1]) else 0
                if lag <= 6:  # Volatility lags for short term
                    vol_24 = df['returns'].rolling(24, min_periods=12).std()
                    features[f'volatility_lag_{lag}'] = vol_24.iloc[-lag-1] if len(vol_24) > lag and not pd.isna(vol_24.iloc[-lag-1]) else 0
        
        # Volume features (important for breakout confirmation)
        volume_sma = df['volume'].rolling(24, min_periods=12).mean()
        if volume_sma.iloc[-1] > 0:
            features['volume_ratio'] = df['volume'].iloc[-1] / volume_sma.iloc[-1]
        
        # Volatility regime features (critical for target-specific models)
        vol_24 = df['returns'].rolling(24, min_periods=12).std().iloc[-1]
        vol_168 = df['returns'].rolling(168, min_periods=84).std().iloc[-1]
        if vol_168 > 0:
            features['volatility_regime'] = vol_24 / vol_168  # Current vs weekly volatility
        
        # Time-based features
        latest_timestamp = df['timestamp'].iloc[-1]
        if isinstance(latest_timestamp, str):
            latest_timestamp = pd.to_datetime(latest_timestamp)
            
        hour = latest_timestamp.hour
        features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        features['is_weekend'] = 1 if latest_timestamp.weekday() >= 5 else 0
        
        # Market structure features
        recent_highs = df['high'].tail(48)  # 48 hour highs
        recent_lows = df['low'].tail(48)    # 48 hour lows
        current_price = df['close'].iloc[-1]
        
        features['distance_to_recent_high'] = (recent_highs.max() - current_price) / current_price
        features['distance_to_recent_low'] = (current_price - recent_lows.min()) / current_price
        
        return features
        
    def create_targets_for_all_percentages(self, df: pd.DataFrame, start_idx: int, 
                                         max_hours: int = 72) -> Dict[float, Optional[int]]:
        """Create targets for all percentage levels"""
        targets = {}
        
        if start_idx >= len(df) - 2:  # Need at least 2 future candles
            return {pct: None for pct in self.target_percentages}
            
        start_price = df['close'].iloc[start_idx]
        
        for target_pct in self.target_percentages:
            target_up = start_price * (1 + target_pct)
            target_down = start_price * (1 - target_pct)
            
            # Look forward to see which target gets hit first
            hit_up = False
            hit_down = False
            
            for j in range(start_idx + 1, min(start_idx + max_hours + 1, len(df))):
                high = df['high'].iloc[j]
                low = df['low'].iloc[j]
                
                if high >= target_up:
                    hit_up = True
                    break
                elif low <= target_down:
                    hit_down = True
                    break
            
            if hit_up:
                targets[target_pct] = 1  # UP
            elif hit_down:
                targets[target_pct] = 0  # DOWN
            else:
                targets[target_pct] = None  # Neither hit
                
        return targets
        
    def prepare_training_datasets(self, df: pd.DataFrame) -> Dict[float, Tuple[pd.DataFrame, pd.Series]]:
        """Prepare separate datasets for each target percentage"""
        if len(df) < 200:  # Need sufficient data
            return {}
            
        datasets = {pct: (None, None) for pct in self.target_percentages}
        
        # Collect features and targets
        feature_list = []
        all_targets = {pct: [] for pct in self.target_percentages}
        
        # Process data (leave buffer for future target calculation)
        start_idx = 168  # 1 week of lookback needed
        end_idx = len(df) - 72  # 72 hours buffer for target calculation
        
        print(f"Processing {end_idx - start_idx} samples for feature engineering...")
        
        for i in range(start_idx, end_idx):
            if i % 100 == 0:
                print(f"Processing sample {i - start_idx + 1}/{end_idx - start_idx}")
                
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
                print(f"Target {target_pct:.1%}: {len(X_final)} samples, distribution: {y_final.value_counts(normalize=True).to_dict()}")
            else:
                print(f"Target {target_pct:.1%}: Insufficient samples ({len(X_final)})")
                
        return datasets
```

### Step 2.1b: Enhanced Feature Sources (Optional)
```python
# src/data/enhanced_features.py
# Example of how to extend beyond CCXT for additional features

import ccxt
import requests
from typing import Dict, Optional
import pandas as pd

class EnhancedDataFetcher:
    """Fetch additional data beyond basic OHLCV"""
    
    def __init__(self):
        # Initialize exchange connections
        self.binance_futures = ccxt.binance({
            'options': {'defaultType': 'future'}
        })
        
    def fetch_open_interest(self, symbol: str) -> Optional[float]:
        """Fetch open interest from Binance futures"""
        try:
            # Binance-specific endpoint for open interest
            ticker = self.binance_futures.fetch_ticker(symbol)
            return ticker.get('openInterest', None)
        except Exception as e:
            logging.warning(f"Could not fetch open interest: {e}")
            return None
            
    def fetch_funding_rate(self, symbol: str) -> Optional[float]:
        """Fetch funding rate for perpetual futures"""
        try:
            # Exchange-specific implementation
            markets = self.binance_futures.load_markets()
            if symbol in markets:
                market = markets[symbol]
                if market.get('type') == 'swap':
                    funding = self.binance_futures.fetch_funding_rate(symbol)
                    return funding.get('fundingRate', None)
        except Exception as e:
            logging.warning(f"Could not fetch funding rate: {e}")
            return None
            
    def fetch_fear_greed_index(self) -> Optional[int]:
        """Fetch crypto fear & greed index"""
        try:
            response = requests.get('https://api.alternative.me/fng/')
            data = response.json()
            return int(data['data'][0]['value'])
        except Exception as e:
            logging.warning(f"Could not fetch fear & greed index: {e}")
            return None
            
    def create_enhanced_features(self, base_features: Dict) -> Dict:
        """Add enhanced features to base feature set"""
        # Example of adding additional features
        enhanced = base_features.copy()
        
        # Add open interest ratio if available
        oi = self.fetch_open_interest('BTC/USDT:USDT')
        if oi:
            enhanced['open_interest_ratio'] = oi / base_features.get('volume', 1)
            
        # Add market sentiment
        fng = self.fetch_fear_greed_index()
        if fng:
            enhanced['fear_greed_index'] = fng / 100.0  # Normalize to 0-1
            
        return enhanced

# Integration with existing feature engineering:
# 1. Use enhanced features when available
# 2. Fall back to OHLCV-only features if APIs fail
# 3. Consider feature importance to decide if enhanced data is worth the complexity
```

**🔍 CHECKPOINT 2A TEST:**
```python
# test_feature_engineering.py
from src.data.feature_engineer import MultiTargetFeatureEngineer
from src.database.operations import DatabaseOperations
from src.utils.config import Config

# Get test data
config = Config()
db_ops = DatabaseOperations(config.database_url)
btc_data = db_ops.get_latest_candles('BTC/USDT', limit=500)

if len(btc_data) >= 200:
    # Test feature engineering
    feature_engineer = MultiTargetFeatureEngineer()
    
    # Test single feature creation
    features = feature_engineer.create_features(btc_data)
    print(f"✅ Created {len(features)} features")
    print("Sample features:", list(features.keys())[:10])
    
    # Test multi-target dataset preparation
    datasets = feature_engineer.prepare_training_datasets(btc_data)
    print(f"✅ Prepared datasets for {len(datasets)} targets")
    
    for target_pct, (X, y) in datasets.items():
        if X is not None:
            print(f"Target {target_pct:.1%}: {len(X)} samples, {len(X.columns)} features")
else:
    print("❌ Need more historical data for testing")
```

---

## CHECKPOINT 2B: Multi-Target Model Training (Day 4-5, 8 hours)

### Step 2.2: Multi-Target Model Trainer
```python
# src/models/multi_target_trainer.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional

class MultiTargetModelTrainer:
    def __init__(self, db_operations, target_percentages: List[float] = [0.01, 0.02, 0.05]):
        self.db = db_operations
        self.target_percentages = target_percentages
        self.models = {}  # {(symbol, target_pct): model_data}
        self.feature_importance = {}
        
    def train_models_for_symbol(self, symbol: str, retrain: bool = False) -> Dict[float, bool]:
        """Train all target models for a symbol"""
        print(f"\n🚀 Training models for {symbol}")
        print("=" * 50)
        
        results = {}
        
        # Get historical data
        df = self.db.get_latest_candles(symbol, '1h', limit=2000)  # ~3 months
        
        if len(df) < 500:
            print(f"❌ Insufficient data for {symbol}: {len(df)} candles")
            return {pct: False for pct in self.target_percentages}
        
        print(f"📊 Using {len(df)} candles from {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Prepare datasets for all targets
        from src.data.feature_engineer import MultiTargetFeatureEngineer
        feature_engineer = MultiTargetFeatureEngineer(self.target_percentages)
        datasets = feature_engineer.prepare_training_datasets(df)
        
        # Train model for each target percentage
        for target_pct in self.target_percentages:
            print(f"\n🎯 Training {target_pct:.1%} target model...")
            
            if target_pct not in datasets or datasets[target_pct][0] is None:
                print(f"❌ No valid dataset for {target_pct:.1%} target")
                results[target_pct] = False
                continue
                
            X, y = datasets[target_pct]
            success = self._train_single_model(symbol, target_pct, X, y)
            results[target_pct] = success
            
        return results
        
    def _train_single_model(self, symbol: str, target_pct: float, X: pd.DataFrame, y: pd.Series) -> bool:
        """Train a single model for specific symbol and target"""
        try:
            print(f"   📈 Dataset: {len(X)} samples, {len(X.columns)} features")
            print(f"   📊 Target distribution: {y.value_counts(normalize=True).round(3).to_dict()}")
            
            # Check if we have balanced enough data
            class_counts = y.value_counts()
            min_class_size = class_counts.min()
            
            if min_class_size < 50:
                print(f"   ❌ Insufficient samples for minority class: {min_class_size}")
                return False
            
            # Configure model based on target percentage
            if target_pct <= 0.01:  # 1% targets - more sensitive to recent patterns
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=8,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1,
                    class_weight='balanced'
                )
            elif target_pct <= 0.02:  # 2% targets - balanced approach
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=8,
                    min_samples_leaf=4,
                    random_state=42,
                    n_jobs=-1,
                    class_weight='balanced'
                )
            else:  # 5% targets - can be deeper, less noise
                model = RandomForestClassifier(
                    n_estimators=150,
                    max_depth=12,
                    min_samples_split=6,
                    min_samples_leaf=3,
                    random_state=42,
                    n_jobs=-1,
                    class_weight='balanced'
                )
            
            # Time series cross-validation
            print(f"   🔄 Cross-validating...")
            tscv = TimeSeriesSplit(n_splits=3)
            cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy', n_jobs=-1)
            
            mean_accuracy = cv_scores.mean()
            std_accuracy = cv_scores.std()
            
            print(f"   📊 CV Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
            
            # Check if model is better than random
            if mean_accuracy < 0.52:  # Must be better than random + small margin
                print(f"   ❌ Model accuracy too low: {mean_accuracy:.4f}")
                return False
            
            # Train final model on all data
            print(f"   🏋️ Training final model...")
            model.fit(X, y)
            
            # Calculate feature importance
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"   🔍 Top 5 features:")
            for idx, row in feature_importance.head().iterrows():
                print(f"      {row['feature']}: {row['importance']:.4f}")
            
            # Save model data
            model_key = (symbol, target_pct)
            self.models[model_key] = {
                'model': model,
                'feature_cols': list(X.columns),
                'accuracy': mean_accuracy,
                'std_accuracy': std_accuracy,
                'target_pct': target_pct,
                'trained_at': datetime.now(),
                'training_samples': len(X)
            }
            
            self.feature_importance[model_key] = feature_importance
            
            # Save to disk
            self._save_model_to_disk(symbol, target_pct, self.models[model_key])
            
            print(f"   ✅ Model trained successfully! Accuracy: {mean_accuracy:.4f}")
            return True
            
        except Exception as e:
            print(f"   ❌ Error training model: {e}")
            logging.error(f"Error training {symbol} {target_pct} model: {e}")
            return False
            
    def _save_model_to_disk(self, symbol: str, target_pct: float, model_data: Dict):
        """Save model to disk"""
        os.makedirs("models", exist_ok=True)
        
        # Clean symbol name for filename
        clean_symbol = symbol.replace('/', '_')
        filename = f"{clean_symbol}_{target_pct:.3f}_model.pkl"
        filepath = f"models/{filename}"
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"   💾 Model saved to {filepath}")
        except Exception as e:
            print(f"   ❌ Error saving model: {e}")
            
    def load_model_from_disk(self, symbol: str, target_pct: float) -> Optional[Dict]:
        """Load model from disk"""
        clean_symbol = symbol.replace('/', '_')
        filename = f"{clean_symbol}_{target_pct:.3f}_model.pkl"
        filepath = f"models/{filename}"
        
        if not os.path.exists(filepath):
            return None
            
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            return model_data
        except Exception as e:
            logging.error(f"Error loading model {filepath}: {e}")
            return None
            
    def train_all_models(self, symbols: List[str], retrain: bool = False) -> Dict[str, Dict[float, bool]]:
        """Train models for all symbols and targets"""
        print("\n🎯 MULTI-TARGET MODEL TRAINING")
        print("=" * 60)
        
        all_results = {}
        
        for symbol in symbols:
            results = self.train_models_for_symbol(symbol, retrain)
            all_results[symbol] = results
            
            # Summary for this symbol
            successful = sum(1 for success in results.values() if success)
            total = len(results)
            print(f"\n📊 {symbol} Summary: {successful}/{total} models trained successfully")
            
        # Overall summary
        print(f"\n🎉 TRAINING COMPLETE")
        print("=" * 60)
        total_models = sum(len(results) for results in all_results.values())
        successful_models = sum(sum(1 for success in results.values() if success) for results in all_results.values())
        print(f"Successfully trained: {successful_models}/{total_models} models")
        
        return all_results
        
    def get_model_summary(self) -> pd.DataFrame:
        """Get summary of all trained models"""
        summary_data = []
        
        for (symbol, target_pct), model_data in self.models.items():
            summary_data.append({
                'symbol': symbol,
                'target_pct': f"{target_pct:.1%}",
                'accuracy': model_data['accuracy'],
                'std_accuracy': model_data['std_accuracy'],
                'training_samples': model_data['training_samples'],
                'trained_at': model_data['trained_at'],
                'features_count': len(model_data['feature_cols'])
            })
            
        if summary_data:
            return pd.DataFrame(summary_data).sort_values(['symbol', 'target_pct'])
        else:
            return pd.DataFrame()
```

**🔍 CHECKPOINT 2B TEST:**
```python
# test_model_training.py
from src.models.multi_target_trainer import MultiTargetModelTrainer
from src.database.operations import DatabaseOperations
from src.utils.config import Config

# Initialize
config = Config()
db_ops = DatabaseOperations(config.database_url)
trainer = MultiTargetModelTrainer(db_ops)

# Test with one symbol first
test_results = trainer.train_models_for_symbol('BTC/USDT')
print("✅ Training results:", test_results)

# Check model summary
summary = trainer.get_model_summary()
if not summary.empty:
    print("✅ Model Summary:")
    print(summary.to_string(index=False))
else:
    print("❌ No models trained successfully")
```

---

## CHECKPOINT 2C: Prediction Engine (Day 6, 6 hours)

### Step 2.3: Multi-Target Prediction Engine
```python
# src/models/multi_target_predictor.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
import json

class MultiTargetPredictionEngine:
    def __init__(self, db_operations, target_percentages: List[float] = [0.01, 0.02, 0.05]):
        self.db = db_operations
        self.target_percentages = target_percentages
        self.trainer = None  # Will be set from outside
        self.feature_engineer = None  # Will be set from outside
        
    def predict_symbol_all_targets(self, symbol: str) -> Dict[str, any]:
        """Make predictions for all targets for a single symbol"""
        try:
            print(f"🔮 Predicting {symbol} for all targets...")
            
            # Get recent data for feature generation
            df = self.db.get_latest_candles(symbol, '1h', limit=200)
            
            if df.empty or len(df) < 50:
                return {
                    'symbol': symbol,
                    'error': f'Insufficient recent data: {len(df)} candles',
                    'predictions': {}
                }
            
            # Generate features
            if self.feature_engineer is None:
                from src.data.feature_engineer import MultiTargetFeatureEngineer
                self.feature_engineer = MultiTargetFeatureEngineer(self.target_percentages)
                
            features = self.feature_engineer.create_features(df)
            
            if features is None:
                return {
                    'symbol': symbol,
                    'error': 'Could not generate features',
                    'predictions': {}
                }
            
            # Make predictions for each target
            predictions = {}
            latest_timestamp = df['timestamp'].iloc[-1]
            
            for target_pct in self.target_percentages:
                pred_result = self._predict_single_target(symbol, target_pct, features, latest_timestamp)
                predictions[f"{target_pct:.1%}"] = pred_result
            
            return {
                'symbol': symbol,
                'timestamp': latest_timestamp.isoformat(),
                'predictions': predictions,
                'features_count': len(features)
            }
            
        except Exception as e:
            logging.error(f"Error predicting {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': f'Prediction failed: {str(e)}',
                'predictions': {}
            }
            
    def _predict_single_target(self, symbol: str, target_pct: float, features: Dict, timestamp: datetime) -> Dict:
        """Make prediction for a single target"""
        try:
            # Load model if not already loaded
            model_key = (symbol, target_pct)
            
            if self.trainer is None:
                from src.models.multi_target_trainer import MultiTargetModelTrainer
                self.trainer = MultiTargetModelTrainer(self.db, self.target_percentages)
            
            if model_key not in self.trainer.models:
                model_data = self.trainer.load_model_from_disk(symbol, target_pct)
                if model_data is None:
                    return {
                        'error': f'No trained model for {target_pct:.1%} target',
                        'model_available': False
                    }
                self.trainer.models[model_key] = model_data
            
            model_data = self.trainer.models[model_key]
            model = model_data['model']
            feature_cols = model_data['feature_cols']
            
            # Prepare feature vector
            feature_vector = []
            missing_features = []
            
            for col in feature_cols:
                if col in features:
                    feature_vector.append(features[col])
                else:
                    feature_vector.append(0)  # Default value for missing features
                    missing_features.append(col)
            
            if len(missing_features) > len(feature_cols) * 0.2:  # More than 20% missing
                return {
                    'error': f'Too many missing features: {len(missing_features)}/{len(feature_cols)}',
                    'missing_features': missing_features[:5]  # Show first 5
                }
            
            # Make prediction
            X_pred = np.array(feature_vector).reshape(1, -1)
            probabilities = model.predict_proba(X_pred)[0]
            
            prediction_class = 1 if probabilities[1] > 0.5 else 0
            confidence = max(probabilities)
            up_probability = probabilities[1]
            down_probability = probabilities[0]
            
            return {
                'prediction': 'UP' if prediction_class == 1 else 'DOWN',
                'up_probability': float(up_probability),
                'down_probability': float(down_probability),
                'confidence': float(confidence),
                'model_accuracy': float(model_data['accuracy']),
                'missing_features_count': len(missing_features),
                'model_available': True
            }
            
        except Exception as e:
            logging.error(f"Error predicting {symbol} {target_pct}: {e}")
            return {
                'error': f'Prediction failed: {str(e)}',
                'model_available': False
            }
            
    def predict_all_assets(self, symbols: List[str]) -> Dict[str, Dict]:
        """Make predictions for all assets and all targets"""
        print("\n🔮 MAKING MULTI-TARGET PREDICTIONS")
        print("=" * 50)
        
        all_predictions = {}
        
        for symbol in symbols:
            prediction_result = self.predict_symbol_all_targets(symbol)
            all_predictions[symbol] = prediction_result
            
            # Print summary for this symbol
            if 'error' not in prediction_result:
                predictions = prediction_result['predictions']
                successful_preds = sum(1 for pred in predictions.values() if 'error' not in pred)
                print(f"   {symbol}: {successful_preds}/{len(predictions)} predictions successful")
            else:
                print(f"   {symbol}: ❌ {prediction_result['error']}")
        
        print(f"\n✅ Prediction round complete for {len(symbols)} assets")
        return all_predictions
        
    def get_prediction_summary(self, predictions: Dict[str, Dict]) -> pd.DataFrame:
        """Create a summary DataFrame of all predictions"""
        summary_data = []
        
        for symbol, symbol_data in predictions.items():
            if 'error' in symbol_data:
                continue
                
            for target_str, pred_data in symbol_data['predictions'].items():
                if 'error' not in pred_data:
                    summary_data.append({
                        'symbol': symbol,
                        'target': target_str,
                        'prediction': pred_data['prediction'],
                        'confidence': pred_data['confidence'],
                        'up_probability': pred_data['up_probability'],
                        'model_accuracy': pred_data.get('model_accuracy', 0),
                        'timestamp': symbol_data['timestamp']
                    })
        
        if summary_data:
            return pd.DataFrame(summary_data)
        else:
            return pd.DataFrame()
```

**🔍 CHECKPOINT 2C TEST:**
```python
# test_predictions.py
from src.models.multi_target_predictor import MultiTargetPredictionEngine
from src.models.multi_target_trainer import MultiTargetModelTrainer
from src.database.operations import DatabaseOperations
from src.utils.config import Config

# Initialize
config = Config()
db_ops = DatabaseOperations(config.database_url)
trainer = MultiTargetModelTrainer(db_ops)
predictor = MultiTargetPredictionEngine(db_ops)
predictor.trainer = trainer

# Load existing models
print("Loading trained models...")
for symbol in ['BTC/USDT']:
    for target_pct in [0.01, 0.02, 0.05]:
        model_data = trainer.load_model_from_disk(symbol, target_pct)
        if model_data:
            trainer.models[(symbol, target_pct)] = model_data
            print(f"✅ Loaded {symbol} {target_pct:.1%} model")

# Test predictions
test_predictions = predictor.predict_all_assets(['BTC/USDT'])
print("✅ Predictions completed")

# Show summary
summary = predictor.get_prediction_summary(test_predictions)
if not summary.empty:
    print("✅ Prediction Summary:")
    print(summary.to_string(index=False))
else:
    print("❌ No successful predictions")
```

---

## Phase 2 Success Criteria

After completing Phase 2, you should have:

✅ **Feature engineering pipeline** that creates 40+ features from price data  
✅ **9 trained ML models** (3 assets × 3 targets) with 60%+ accuracy  
✅ **Prediction engine** that outputs probabilities for each target  
✅ **Model persistence** system that saves/loads trained models  
✅ **Performance tracking** with feature importance analysis  

**Next Step**: Move to Phase 3 to add Telegram integration for automated reporting.
