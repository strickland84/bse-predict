"""Optimized model trainer with hyperparameter tuning and advanced feature engineering."""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import (
    RandomizedSearchCV, GridSearchCV, TimeSeriesSplit, 
    cross_val_score, StratifiedKFold
)
from sklearn.metrics import (
    classification_report, accuracy_score, precision_recall_fscore_support,
    make_scorer, f1_score, matthews_corrcoef, confusion_matrix
)
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from scipy.stats import randint, uniform
import pickle
import os
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from src.database.operations import db_ops
from src.data.feature_engineer import MultiTargetFeatureEngineer
from src.utils.logger import get_logger
from src.utils.config import config
from sqlalchemy import text
import json

logger = get_logger(__name__)


class OptimizedModelTrainer:
    """Advanced model trainer with hyperparameter optimization and feature engineering."""
    
    def __init__(self, target_percentages: List[float] = None):
        """Initialize optimized trainer.
        
        Args:
            target_percentages: List of target percentages [0.01, 0.02, 0.05]
        """
        self.target_percentages = target_percentages or config.target_percentages
        self.models = {}
        self.best_params = {}
        self.feature_importance = {}
        self.models_dir = Path("models_optimized")
        self.models_dir.mkdir(exist_ok=True)
        
    def get_hyperparameter_space(self, model_type: str, target_pct: float) -> Dict[str, Any]:
        """Get hyperparameter search space based on model type and target.
        
        Args:
            model_type: Type of model ('rf', 'xgb', 'lgb', 'gb')
            target_pct: Target percentage
            
        Returns:
            Dictionary of hyperparameter distributions
        """
        if model_type == 'rf':
            # RandomForest hyperparameters
            base_params = {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [3, 5, 7, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4, 8],
                'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7],
                'bootstrap': [True, False],
                'class_weight': ['balanced', 'balanced_subsample', None],
                'random_state': [42]
            }
            
            # Adjust for different targets
            if target_pct <= 0.01:  # 1% - need more sensitivity
                base_params['max_depth'] = [3, 5, 7, 10]
                base_params['min_samples_leaf'] = [4, 8, 12]
            elif target_pct <= 0.02:  # 2% - balanced
                base_params['max_depth'] = [5, 7, 10, 15]
            else:  # 5% - can be deeper
                base_params['max_depth'] = [7, 10, 15, 20, None]
                
        elif model_type == 'xgb':
            # XGBoost hyperparameters
            base_params = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.3],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'gamma': [0, 0.1, 0.5, 1],
                'reg_alpha': [0, 0.1, 1, 10],
                'reg_lambda': [0, 0.1, 1, 10],
                'scale_pos_weight': [1, 2, 3],  # For imbalanced classes
                'random_state': [42]
            }
            
        elif model_type == 'lgb':
            # LightGBM hyperparameters
            base_params = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7, 10, -1],
                'learning_rate': [0.01, 0.05, 0.1, 0.3],
                'num_leaves': [20, 31, 50, 100],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'reg_alpha': [0, 0.1, 1],
                'reg_lambda': [0, 0.1, 1],
                'min_child_samples': [5, 10, 20],
                'is_unbalance': [True, False],
                'random_state': [42],
                'verbosity': [-1]
            }
            
        elif model_type == 'gb':
            # GradientBoosting hyperparameters
            base_params = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.3],
                'subsample': [0.6, 0.8, 1.0],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', 0.3, 0.5],
                'random_state': [42]
            }
            
        return base_params
    
    def engineer_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer advanced features from OHLCV and futures data.
        
        Args:
            df: DataFrame with OHLCV and futures data
            
        Returns:
            DataFrame with additional engineered features
        """
        df = df.copy()
        
        # Price-based features
        df['price_change'] = df['close'].pct_change()
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # Volume features
        df['volume_sma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['volume_ema_ratio'] = df['volume'] / df['volume'].ewm(span=20).mean()
        
        # Volatility features
        df['true_range'] = df[['high', 'close']].max(axis=1) - df[['low', 'close']].min(axis=1)
        df['atr_14'] = df['true_range'].rolling(14).mean()
        df['volatility'] = df['close'].pct_change().rolling(20).std()
        
        # Additional technical indicators
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']
        
        # Stochastic RSI
        rsi = self.calculate_rsi(df['close'], 14)
        stoch_rsi = (rsi - rsi.rolling(14).min()) / (rsi.rolling(14).max() - rsi.rolling(14).min())
        df['stoch_rsi'] = stoch_rsi
        
        # Price position in range
        df['price_position'] = (df['close'] - df['low'].rolling(20).min()) / \
                               (df['high'].rolling(20).max() - df['low'].rolling(20).min())
        
        # Futures-specific features (if available)
        if 'open_interest' in df.columns:
            df['oi_change'] = df['open_interest'].pct_change()
            df['oi_volume_ratio'] = df['open_interest'] / df['volume']
            df['oi_sma_ratio'] = df['open_interest'] / df['open_interest'].rolling(20).mean()
            
        if 'funding_rate' in df.columns:
            df['funding_cumsum'] = df['funding_rate'].rolling(24).sum()  # 24h cumulative
            df['funding_ma'] = df['funding_rate'].rolling(8).mean()  # 8h MA
            df['funding_std'] = df['funding_rate'].rolling(24).std()
            
        # Interaction features
        df['rsi_volume'] = rsi * df['volume_sma_ratio']
        df['macd_volatility'] = df['macd_diff'] * df['volatility']
        
        # Lag features for time series
        for lag in [1, 3, 6, 12, 24]:
            df[f'return_lag_{lag}'] = df['price_change'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume_sma_ratio'].shift(lag)
            
        return df
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def perform_feature_selection(self, X: pd.DataFrame, y: pd.Series, 
                                 method: str = 'kbest', n_features: int = 30) -> List[str]:
        """Select best features using various methods.
        
        Args:
            X: Feature matrix
            y: Target vector
            method: Selection method ('kbest', 'rfe')
            n_features: Number of features to select
            
        Returns:
            List of selected feature names
        """
        if method == 'kbest':
            selector = SelectKBest(f_classif, k=min(n_features, len(X.columns)))
            selector.fit(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            
        elif method == 'rfe':
            estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            selector = RFE(estimator, n_features_to_select=min(n_features, len(X.columns)))
            selector.fit(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            
        else:
            selected_features = X.columns.tolist()
            
        return selected_features
    
    def optimize_model(self, X: pd.DataFrame, y: pd.Series, symbol: str, 
                      target_pct: float, model_type: str = 'rf',
                      search_type: str = 'random', cv_folds: int = 5) -> Tuple[Any, Dict, float]:
        """Optimize model hyperparameters using grid or random search.
        
        Args:
            X: Feature matrix
            y: Target vector
            symbol: Trading pair symbol
            target_pct: Target percentage
            model_type: Type of model to optimize
            search_type: 'grid' or 'random'
            cv_folds: Number of CV folds
            
        Returns:
            Tuple of (best_model, best_params, best_score)
        """
        print(f"\n🔧 Optimizing {model_type.upper()} model for {symbol} {target_pct:.1%}")
        
        # Get hyperparameter space
        param_space = self.get_hyperparameter_space(model_type, target_pct)
        
        # Create base model
        if model_type == 'rf':
            base_model = RandomForestClassifier(n_jobs=-1)
        elif model_type == 'xgb':
            base_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=-1)
        elif model_type == 'lgb':
            base_model = lgb.LGBMClassifier(n_jobs=-1)
        elif model_type == 'gb':
            base_model = GradientBoostingClassifier()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create CV strategy
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        # Custom scoring with F1 weighted
        scoring = make_scorer(f1_score, average='weighted')
        
        # Perform search
        if search_type == 'grid':
            search = GridSearchCV(
                base_model, param_space, cv=tscv, scoring=scoring,
                n_jobs=-1, verbose=1, return_train_score=True
            )
        else:  # random search
            search = RandomizedSearchCV(
                base_model, param_space, n_iter=50, cv=tscv, scoring=scoring,
                n_jobs=-1, verbose=1, random_state=42, return_train_score=True
            )
        
        # Fit search
        print(f"   🔍 Running {search_type} search with {cv_folds}-fold CV...")
        search.fit(X, y)
        
        print(f"   ✅ Best score: {search.best_score_:.4f}")
        print(f"   📊 Best params: {search.best_params_}")
        
        return search.best_estimator_, search.best_params_, search.best_score_
    
    def train_optimized_model(self, symbol: str, target_pct: float, 
                            model_type: str = 'rf', optimize: bool = True) -> bool:
        """Train an optimized model for a specific symbol and target.
        
        Args:
            symbol: Trading pair symbol
            target_pct: Target percentage
            model_type: Model type to use
            optimize: Whether to perform hyperparameter optimization
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"\n🚀 Training optimized {model_type.upper()} model for {symbol} {target_pct:.1%}")
            
            # Get data with futures
            training_limit = config.get_nested('ml.training_data_limit', 4000)
            df = db_ops.get_latest_candles(symbol, '1h', limit=training_limit, include_futures=True)
            
            if df.empty or len(df) < 500:
                print(f"❌ Insufficient data: {len(df)} candles")
                return False
            
            print(f"📊 Loaded {len(df)} candles with futures data")
            
            # Engineer advanced features
            print("   🔬 Engineering advanced features...")
            df = self.engineer_advanced_features(df)
            
            # Prepare dataset
            feature_engineer = MultiTargetFeatureEngineer([target_pct])
            datasets = feature_engineer.prepare_training_datasets(df)
            
            if target_pct not in datasets or datasets[target_pct][0] is None:
                print(f"❌ Failed to prepare dataset")
                return False
            
            X, y = datasets[target_pct]
            print(f"   📈 Dataset: {len(X)} samples, {len(X.columns)} features")
            
            # Feature selection
            print("   🎯 Performing feature selection...")
            selected_features = self.perform_feature_selection(X, y, method='kbest', n_features=40)
            X_selected = X[selected_features]
            print(f"   ✅ Selected {len(selected_features)} best features")
            
            # Optimize or use default params
            if optimize:
                best_model, best_params, best_score = self.optimize_model(
                    X_selected, y, symbol, target_pct, model_type, search_type='random'
                )
            else:
                # Use reasonable defaults
                if model_type == 'rf':
                    best_model = RandomForestClassifier(
                        n_estimators=200, max_depth=10, min_samples_split=5,
                        class_weight='balanced', random_state=42, n_jobs=-1
                    )
                elif model_type == 'xgb':
                    best_model = xgb.XGBClassifier(
                        n_estimators=200, max_depth=7, learning_rate=0.1,
                        scale_pos_weight=2, random_state=42, n_jobs=-1
                    )
                best_model.fit(X_selected, y)
                best_params = best_model.get_params()
                best_score = 0.0
            
            # Final evaluation
            split_idx = int(len(X_selected) * 0.8)
            X_train, X_test = X_selected.iloc[:split_idx], X_selected.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Retrain on full training set
            best_model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
            mcc = matthews_corrcoef(y_test, y_pred)
            
            print(f"\n   📊 Test Set Performance:")
            print(f"      Accuracy: {accuracy:.4f}")
            print(f"      Precision: {precision:.4f}")
            print(f"      Recall: {recall:.4f}")
            print(f"      F1-Score: {f1:.4f}")
            print(f"      MCC: {mcc:.4f}")
            
            # Feature importance
            if hasattr(best_model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': selected_features,
                    'importance': best_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                print(f"\n   🔍 Top 10 Features:")
                for _, row in feature_importance.head(10).iterrows():
                    print(f"      {row['feature']}: {row['importance']:.4f}")
            
            # Save model
            model_key = (symbol, target_pct, model_type)
            self.models[model_key] = {
                'model': best_model,
                'feature_cols': selected_features,
                'params': best_params,
                'cv_score': best_score,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'mcc': mcc,
                'trained_at': datetime.now(),
                'model_type': model_type
            }
            
            # Save to disk
            self._save_optimized_model(symbol, target_pct, model_type, self.models[model_key])
            
            print(f"   ✅ Model trained and saved successfully!")
            return True
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            logger.error(f"Error training optimized model: {e}")
            return False
    
    def _save_optimized_model(self, symbol: str, target_pct: float, model_type: str, model_data: Dict):
        """Save optimized model to disk."""
        clean_symbol = symbol.replace('/', '_')
        filename = f"{model_type}_{clean_symbol}_{target_pct:.3f}_optimized.pkl"
        filepath = self.models_dir / filename
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"   💾 Saved to {filepath}")
    
    def compare_models(self, symbol: str, target_pct: float, 
                      model_types: List[str] = ['rf', 'xgb', 'lgb']) -> pd.DataFrame:
        """Train and compare multiple model types.
        
        Args:
            symbol: Trading pair symbol
            target_pct: Target percentage
            model_types: List of model types to compare
            
        Returns:
            DataFrame with comparison results
        """
        results = []
        
        for model_type in model_types:
            print(f"\n{'='*60}")
            print(f"Training {model_type.upper()} model...")
            
            success = self.train_optimized_model(symbol, target_pct, model_type, optimize=True)
            
            if success:
                model_key = (symbol, target_pct, model_type)
                if model_key in self.models:
                    model_data = self.models[model_key]
                    results.append({
                        'model_type': model_type,
                        'cv_score': model_data.get('cv_score', 0),
                        'accuracy': model_data['accuracy'],
                        'precision': model_data['precision'],
                        'recall': model_data['recall'],
                        'f1_score': model_data['f1_score'],
                        'mcc': model_data.get('mcc', 0)
                    })
        
        if results:
            comparison_df = pd.DataFrame(results).sort_values('f1_score', ascending=False)
            print(f"\n📊 Model Comparison for {symbol} {target_pct:.1%}:")
            print(comparison_df.to_string())
            return comparison_df
        
        return pd.DataFrame()
    
    def train_all_optimized(self, symbols: List[str] = None, 
                           model_type: str = 'xgb') -> Dict[str, Dict[float, bool]]:
        """Train optimized models for all symbols and targets.
        
        Args:
            symbols: List of symbols (default from config)
            model_type: Model type to use
            
        Returns:
            Training results
        """
        if symbols is None:
            symbols = config.assets
        
        print(f"\n🎯 OPTIMIZED MODEL TRAINING - {model_type.upper()}")
        print("=" * 60)
        
        all_results = {}
        
        for symbol in symbols:
            symbol_results = {}
            for target_pct in self.target_percentages:
                success = self.train_optimized_model(symbol, target_pct, model_type, optimize=True)
                symbol_results[target_pct] = success
            all_results[symbol] = symbol_results
        
        # Summary
        print(f"\n🎉 TRAINING COMPLETE")
        print("=" * 60)
        for symbol, results in all_results.items():
            successful = sum(1 for s in results.values() if s)
            print(f"{symbol}: {successful}/{len(results)} models trained")
        
        return all_results