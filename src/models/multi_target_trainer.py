"""Multi-target model training for cryptocurrency price prediction."""
import pandas as pd
import numpy as np
import warnings
# Suppress sklearn warnings about constant features
warnings.filterwarnings('ignore', message='Features .* are constant', category=UserWarning)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import pickle
import os
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from src.database.operations import db_ops
from src.data.feature_engineer import MultiTargetFeatureEngineer
from src.utils.logger import get_logger
from src.utils.config import config
from sqlalchemy import text
import json

logger = get_logger(__name__)


class MultiTargetModelTrainer:
    """Handles training and management of multiple ML models for different targets."""
    
    def __init__(self, target_percentages: List[float] = None):
        """Initialize model trainer.
        
        Args:
            target_percentages: List of target percentages [0.01, 0.02, 0.05]
        """
        self.target_percentages = target_percentages or config.target_percentages
        self.models = {}  # {(symbol, target_pct): model_data}
        self.feature_importance = {}
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
    def train_models_for_symbol(self, symbol: str, retrain: bool = False) -> Dict[float, bool]:
        """Train all target models for a single symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            retrain: Whether to retrain existing models
            
        Returns:
            Dictionary mapping target percentages to success status
        """
        print(f"\n🚀 Training models for {symbol}")
        print("=" * 50)
        
        results = {}
        
        # Get historical data
        training_limit = config.get_nested('ml.training_data_limit', 4000)
        df = db_ops.get_latest_candles(symbol, '1h', limit=training_limit, include_futures=True)
        
        if df.empty or len(df) < 500:
            print(f"❌ Insufficient data for {symbol}: {len(df)} candles (need 500+)")
            return {pct: False for pct in self.target_percentages}
        
        print(f"📊 Loaded {len(df)} candles for {symbol}")
        print(f"   📅 Date range: {df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}")
        print(f"   💰 Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        
        # Prepare datasets for all targets
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
            success = self._train_single_model(symbol, target_pct, X, y, df, retrain)
            results[target_pct] = success
            
        return results
        
    def _train_single_model(self, symbol: str, target_pct: float, X: pd.DataFrame, y: pd.Series, 
                          df: pd.DataFrame, retrain: bool = False) -> bool:
        """Train a single model for specific symbol and target.
        
        Args:
            symbol: Trading pair symbol
            target_pct: Target percentage
            X: Feature matrix
            y: Target vector
            df: Original dataframe (for metadata)
            retrain: Whether to retrain existing model
            
        Returns:
            True if training successful, False otherwise
        """
        try:
            print(f"   📈 Dataset: {len(X)} samples, {len(X.columns)} features")
            print(f"   📊 Target distribution: {y.value_counts(normalize=True).round(3).to_dict()}")
            
            # Check if we have balanced enough data
            class_counts = y.value_counts()
            min_class_size = class_counts.min()
            
            if min_class_size < 50:
                print(f"   ❌ Insufficient samples for minority class: {min_class_size}")
                return False
            
            # Skip if model already exists and not retraining
            model_path = self._get_model_path(symbol, target_pct)
            if model_path.exists() and not retrain:
                print(f"   ⏭️ Model already exists, skipping (use retrain=True to override)")
                return True
            
            # Configure model based on target percentage
            if target_pct <= 0.01:  # 1% targets - more sensitive to recent patterns
                model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=8,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1,
                    class_weight='balanced'
                )
            elif target_pct <= 0.02:  # 2% targets - balanced approach
                model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_split=8,
                    min_samples_leaf=4,
                    random_state=42,
                    n_jobs=-1,
                    class_weight='balanced'
                )
            else:  # 5% targets - can be deeper, less noise
                model = RandomForestClassifier(
                    n_estimators=300,
                    max_depth=12,
                    min_samples_split=6,
                    min_samples_leaf=3,
                    random_state=42,
                    n_jobs=-1,
                    class_weight='balanced'
                )
            
            # Time series cross-validation
            print(f"   🔄 Cross-validating...")
            tscv = TimeSeriesSplit(n_splits=5)
            cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy', n_jobs=-1)
            
            mean_accuracy = cv_scores.mean()
            std_accuracy = cv_scores.std()
            
            print(f"   📊 CV Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
            
            # Check if model is better than random
            if mean_accuracy < 0.52:  # Must be better than random + small margin
                print(f"   ❌ Model accuracy too low: {mean_accuracy:.4f}")
                return False
            
            # Split data for final evaluation (80/20 split, preserving time order)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Train final model on training portion
            print(f"   🏋️ Training final model on {len(X_train)} samples...")
            model.fit(X_train, y_train)
            
            # Calculate comprehensive metrics on TEST set
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
            
            print(f"   📊 Test set evaluation ({len(X_test)} samples):")
            print(f"      Accuracy: {accuracy:.4f}")
            print(f"      Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            
            # Calculate feature importance
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"   🔍 Top 10 features:")
            for idx, row in feature_importance.head(10).iterrows():
                print(f"      {row['feature']}: {row['importance']:.4f}")
            
            # Save model data
            model_key = (symbol, target_pct)
            self.models[model_key] = {
                'model': model,
                'feature_cols': list(X.columns),
                'accuracy': accuracy,
                'cv_accuracy': mean_accuracy,
                'cv_std': std_accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'target_pct': target_pct,
                'trained_at': datetime.now(),
                'training_samples': len(X),
                'class_distribution': y.value_counts().to_dict()
            }
            
            self.feature_importance[model_key] = feature_importance
            
            # Save to disk
            self._save_model_to_disk(symbol, target_pct, self.models[model_key])
            
            # Save training history to database
            self._save_training_history_to_db(symbol, target_pct, self.models[model_key], df, feature_importance)
            
            print(f"   ✅ Model trained successfully! Accuracy: {accuracy:.4f}")
            return True
            
        except Exception as e:
            print(f"   ❌ Error training model: {e}")
            logger.error(f"Error training {symbol} {target_pct} model: {e}")
            return False
            
    def _save_model_to_disk(self, symbol: str, target_pct: float, model_data: Dict):
        """Save model to disk with versioning.
        
        Args:
            symbol: Trading pair symbol
            target_pct: Target percentage
            model_data: Model data dictionary
        """
        try:
            # Clean symbol name for filename
            clean_symbol = symbol.replace('/', '_').replace('-', '_')
            filename = f"rf_{clean_symbol}_{target_pct:.3f}_model.pkl"
            filepath = self.models_dir / filename
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"   💾 Model saved to {filepath}")
            
        except Exception as e:
            print(f"   ❌ Error saving model: {e}")
            logger.error(f"Error saving model {symbol} {target_pct}: {e}")
            
    def _save_training_history_to_db(self, symbol: str, target_pct: float, model_data: Dict, 
                                   df: pd.DataFrame, feature_importance: pd.DataFrame):
        """Save training history to database.
        
        Args:
            symbol: Trading pair symbol
            target_pct: Target percentage
            model_data: Model data dictionary
            df: Training dataframe (for date and price ranges)
            feature_importance: Feature importance dataframe
        """
        try:
            # Get model configuration based on target percentage
            if target_pct <= 0.01:
                model_config = {
                    "type": "RandomForestClassifier",
                    "n_estimators": 200,
                    "max_depth": 8,
                    "min_samples_split": 10,
                    "min_samples_leaf": 5
                }
            elif target_pct <= 0.02:
                model_config = {
                    "type": "RandomForestClassifier", 
                    "n_estimators": 200,
                    "max_depth": 10,
                    "min_samples_split": 8,
                    "min_samples_leaf": 4
                }
            else:
                model_config = {
                    "type": "RandomForestClassifier",
                    "n_estimators": 300,
                    "max_depth": 12,
                    "min_samples_split": 6,
                    "min_samples_leaf": 3
                }
                
            # Prepare top features (top 10)
            top_features = feature_importance.head(10).to_dict('records')
            
            # Clean symbol for filename
            clean_symbol = symbol.replace('/', '_').replace('-', '_')
            filename = f"rf_{clean_symbol}_{target_pct:.3f}_model.pkl"
            
            query = text("""
                INSERT INTO model_training_history 
                (symbol, target_pct, trained_at, model_filename, training_samples, features_count,
                 date_range_start, date_range_end, price_range_min, price_range_max,
                 target_distribution, cv_accuracy, cv_std, final_accuracy, precision, recall, f1_score,
                 top_features, model_config)
                VALUES (:symbol, :target_pct, :trained_at, :model_filename, :training_samples, :features_count,
                        :date_range_start, :date_range_end, :price_range_min, :price_range_max,
                        :target_distribution, :cv_accuracy, :cv_std, :final_accuracy, :precision, :recall, :f1_score,
                        :top_features, :model_config)
            """)
            
            with db_ops.db.get_session() as session:
                session.execute(query, {
                    'symbol': symbol,
                    'target_pct': float(target_pct),
                    'trained_at': model_data['trained_at'],
                    'model_filename': filename,
                    'training_samples': int(model_data['training_samples']),
                    'features_count': int(len(model_data['feature_cols'])),
                    'date_range_start': df['timestamp'].min(),
                    'date_range_end': df['timestamp'].max(),
                    'price_range_min': float(df['close'].min()),
                    'price_range_max': float(df['close'].max()),
                    'target_distribution': json.dumps(model_data['class_distribution']),
                    'cv_accuracy': float(model_data['cv_accuracy']),
                    'cv_std': float(model_data['cv_std']),
                    'final_accuracy': float(model_data['accuracy']),
                    'precision': float(model_data['precision']),
                    'recall': float(model_data['recall']),
                    'f1_score': float(model_data['f1_score']),
                    'top_features': json.dumps(top_features),
                    'model_config': json.dumps(model_config)
                })
                session.commit()
                
            logger.info(f"Training history saved for {symbol} {target_pct:.1%} model")
            
        except Exception as e:
            logger.error(f"Error saving training history: {e}")
            raise RuntimeError(f"Failed to save model training history: {e}") from e
            
    def load_model_from_disk(self, symbol: str, target_pct: float) -> Optional[Dict]:
        """Load model from disk.
        
        Args:
            symbol: Trading pair symbol
            target_pct: Target percentage
            
        Returns:
            Model data dictionary or None if not found
        """
        try:
            clean_symbol = symbol.replace('/', '_').replace('-', '_')
            filename = f"rf_{clean_symbol}_{target_pct:.3f}_model.pkl"
            filepath = self.models_dir / filename
            
            if not filepath.exists():
                return None
                
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
                
            return model_data
            
        except Exception as e:
            logger.error(f"Error loading model {symbol} {target_pct}: {e}")
            return None
            
    def train_all_models(self, symbols: List[str] = None, retrain: bool = False) -> Dict[str, Dict[float, bool]]:
        """Train models for all symbols and targets.
        
        Args:
            symbols: List of symbols to train. If None, uses config assets.
            retrain: Whether to retrain existing models
            
        Returns:
            Dictionary mapping symbols to training results
        """
        if symbols is None:
            symbols = config.assets
            
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
        """Get summary of all trained models.
        
        Returns:
            DataFrame with model performance metrics
        """
        summary_data = []
        
        for (symbol, target_pct), model_data in self.models.items():
            summary_data.append({
                'symbol': symbol,
                'target_pct': f"{target_pct:.1%}",
                'accuracy': model_data['accuracy'],
                'cv_accuracy': model_data['cv_accuracy'],
                'cv_std': model_data['cv_std'],
                'precision': model_data['precision'],
                'recall': model_data['recall'],
                'f1_score': model_data['f1_score'],
                'training_samples': model_data['training_samples'],
                'trained_at': model_data['trained_at'],
                'features_count': len(model_data['feature_cols'])
            })
            
        if summary_data:
            return pd.DataFrame(summary_data).sort_values(['symbol', 'target_pct'])
        else:
            return pd.DataFrame()
            
    def get_feature_importance_summary(self, symbol: str, target_pct: float) -> Optional[pd.DataFrame]:
        """Get feature importance for a specific model.
        
        Args:
            symbol: Trading pair symbol
            target_pct: Target percentage
            
        Returns:
            DataFrame with feature importance or None if model not found
        """
        model_key = (symbol, target_pct)
        
        if model_key not in self.feature_importance:
            # Try to load from disk
            model_data = self.load_model_from_disk(symbol, target_pct)
            if model_data and 'feature_cols' in model_data:
                # Feature importance would need to be recalculated or stored
                return None
                
        if model_key in self.feature_importance:
            return self.feature_importance[model_key]
            
        return None
        
    def _get_model_path(self, symbol: str, target_pct: float) -> Path:
        """Get model file path.
        
        Args:
            symbol: Trading pair symbol
            target_pct: Target percentage
            
        Returns:
            Path object for model file
        """
        clean_symbol = symbol.replace('/', '_').replace('-', '_')
        filename = f"rf_{clean_symbol}_{target_pct:.3f}_model.pkl"
        return self.models_dir / filename
        
    def validate_models(self, symbols: List[str] = None) -> Dict[str, Dict[float, bool]]:
        """Validate that all required models exist and are loadable.
        
        Args:
            symbols: List of symbols to validate
            
        Returns:
            Dictionary mapping symbols to validation results
        """
        if symbols is None:
            symbols = config.assets
            
        validation_results = {}
        
        for symbol in symbols:
            symbol_results = {}
            for target_pct in self.target_percentages:
                model_data = self.load_model_from_disk(symbol, target_pct)
                symbol_results[target_pct] = model_data is not None
            validation_results[symbol] = symbol_results
            
        return validation_results
        
    def get_training_history(self, limit: int = 20) -> pd.DataFrame:
        """Get model training history from database.
        
        Args:
            limit: Number of records to retrieve
            
        Returns:
            DataFrame with training history
        """
        query = """
            SELECT 
                symbol, target_pct, trained_at, model_filename,
                training_samples, features_count, 
                date_range_start, date_range_end,
                price_range_min, price_range_max,
                cv_accuracy, cv_std, final_accuracy,
                precision, recall, f1_score,
                top_features, model_config,
                created_at
            FROM model_training_history 
            ORDER BY trained_at DESC 
            LIMIT :limit
        """
        
        try:
            with db_ops.db.get_session() as session:
                result = session.execute(text(query), {'limit': limit})
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                
                if not df.empty:
                    # Format the dataframe for display
                    df['target_pct'] = df['target_pct'].apply(lambda x: f"{float(x):.1%}")
                    df['trained_at'] = pd.to_datetime(df['trained_at']).dt.strftime('%Y-%m-%d %H:%M')
                    df['cv_score'] = df.apply(lambda r: f"{r['cv_accuracy']:.3f}±{r['cv_std']:.3f}", axis=1)
                    
                return df
                
        except Exception as e:
            logger.error(f"Error retrieving training history: {e}")
            return pd.DataFrame()
