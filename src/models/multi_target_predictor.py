"""Multi-target prediction engine for cryptocurrency price prediction."""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
import json
from pathlib import Path
from sqlalchemy import text
import pickle

from src.database.operations import db_ops
from src.database.prediction_tracker import prediction_tracker
from src.data.feature_engineer import MultiTargetFeatureEngineer
from src.models.multi_target_trainer import MultiTargetModelTrainer
from src.utils.logger import get_logger
from src.utils.config import config

logger = get_logger(__name__)


class MultiTargetPredictionEngine:
    """Real-time prediction engine for multi-target price movements."""
    
    def __init__(self, target_percentages: List[float] = None, save_predictions: bool = True):
        """Initialize prediction engine.
        
        Args:
            target_percentages: List of target percentages [0.01, 0.02, 0.05]
            save_predictions: Whether to save predictions to database
        """
        self.target_percentages = target_percentages or config.target_percentages
        self.feature_engineer = MultiTargetFeatureEngineer(self.target_percentages)
        self.trainer = MultiTargetModelTrainer(self.target_percentages)
        self.models = {}  # Cache loaded models
        self.save_predictions = save_predictions
        self.use_optimized = bool(config.get_nested('ml.use_optimized_in_predictions', False))
        
    def predict_symbol_all_targets(self, symbol: str) -> Dict[str, any]:
        """Make predictions for all targets for a single symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            
        Returns:
            Dictionary with predictions for all targets
        """
        try:
            logger.info(f"🔮 Predicting {symbol} for all targets...")
            
            # Get recent data for feature generation (need 168h lookback + current)
            df = db_ops.get_latest_candles(symbol, '1h', limit=200, include_futures=True)
            
            if df.empty or len(df) < 168:
                return {
                    'symbol': symbol,
                    'error': f'Insufficient recent data: {len(df)} candles (need 168+)',
                    'predictions': {}
                }
            
            # Generate features using same window as training (last 168 hours + current)
            # This ensures consistency with how features were calculated during training
            window_df = df.iloc[-169:].copy()  # Last 168 hours + current hour
            features = self.feature_engineer.create_features(window_df)
            
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
            logger.error(f"Error predicting {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': f'Prediction failed: {str(e)}',
                'predictions': {}
            }
            
    def _load_active_optimized_model(self, symbol: str, target_pct: float) -> Optional[Dict]:
        """Load active optimized model artifact from model_registry if available."""
        try:
            with db_ops.db.get_session() as session:
                row = session.execute(
                    text("""
                        SELECT file_path
                        FROM model_registry
                        WHERE symbol = :symbol
                          AND target_pct = :target_pct
                          AND is_active = TRUE
                        ORDER BY trained_at DESC
                        LIMIT 1
                    """),
                    {"symbol": symbol, "target_pct": float(target_pct)},
                ).fetchone()
            if not row:
                return None
            file_path = row.file_path if hasattr(row, "file_path") else row[0]
            if not file_path:
                return None
            fp = Path(file_path)
            if not fp.exists():
                return None
            with open(fp, "rb") as f:
                artifact = pickle.load(f)
            return artifact
        except Exception as e:
            logger.error(f"Failed to load optimized model for {symbol} {target_pct}: {e}")
            return None

    def _predict_single_target(self, symbol: str, target_pct: float, features: Dict, 
                             timestamp: datetime) -> Dict:
        """Make prediction for a single target.
        
        Args:
            symbol: Trading pair symbol
            target_pct: Target percentage
            features: Feature dictionary
            timestamp: Prediction timestamp
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Load model if not already loaded
            model_data = None

            # Try optimized active model first if enabled
            if self.use_optimized:
                opt_key = (symbol, target_pct, 'opt')
                if opt_key not in self.models:
                    artifact = self._load_active_optimized_model(symbol, target_pct)
                    if artifact:
                        self.models[opt_key] = artifact
                if opt_key in self.models:
                    model_data = self.models[opt_key]

            # Fallback to baseline model on disk
            if model_data is None:
                base_key = (symbol, target_pct)
                if base_key not in self.models:
                    baseline = self.trainer.load_model_from_disk(symbol, target_pct)
                    if baseline is None:
                        return {
                            'error': f'No trained model for {target_pct:.1%} target',
                            'model_available': False
                        }
                    self.models[base_key] = baseline
                model_data = self.models[base_key]

            # Unify interface
            if 'pipeline' in model_data:
                model = model_data['pipeline']
                feature_cols = model_data.get('feature_columns') or model_data.get('feature_cols')
                used_model_name = f"{model_data.get('model_type', 'model')}_optuna_{target_pct:.1%}"
            else:
                model = model_data['model']
                feature_cols = model_data['feature_cols']
                used_model_name = f"RandomForest_{target_pct:.1%}"
            
            # Prepare feature vector with better missing value handling
            feature_vector = []
            missing_features = []
            
            for col in feature_cols:
                if col in features:
                    feature_vector.append(features[col])
                else:
                    # Use more appropriate defaults based on feature type
                    if 'ratio' in col or 'position' in col or 'momentum' in col:
                        default_value = 0.0  # Neutral for ratios/positions
                    elif 'volume' in col:
                        default_value = 1.0  # Neutral volume ratio
                    elif 'rsi' in col:
                        default_value = 50.0  # Neutral RSI
                    elif 'volatility' in col:
                        default_value = 0.02  # Average volatility
                    else:
                        default_value = 0.0  # Conservative default
                    
                    feature_vector.append(default_value)
                    missing_features.append(col)
            
            # Allow up to 20% missing features
            if len(missing_features) > len(feature_cols) * 0.2:
                return {
                    'error': f'Too many missing features: {len(missing_features)}/{len(feature_cols)}',
                    'missing_features': missing_features[:5]  # Show first 5
                }
            
            # Make prediction - create DataFrame with proper column names
            X_pred = pd.DataFrame([feature_vector], columns=feature_cols)
            probabilities = model.predict_proba(X_pred)[0]
            
            prediction_class = 1 if probabilities[1] > 0.5 else 0
            confidence = max(probabilities)
            up_probability = probabilities[1]
            down_probability = probabilities[0]
            
            # Calculate signal strength
            signal_strength = abs(up_probability - down_probability)
            
            # Save prediction to database if enabled
            if self.save_predictions:
                try:
                    prediction_id = prediction_tracker.save_prediction(
                        symbol=symbol,
                        target_pct=target_pct,
                        timestamp=timestamp,
                        model_name=used_model_name,
                        prediction_class=prediction_class,
                        probability=float(up_probability),
                        confidence=float(confidence),
                        features_used=features
                    )
                    
                    if prediction_id:
                        logger.debug(f"Saved prediction {prediction_id} for {symbol} {target_pct:.1%}")
                except Exception as e:
                    logger.error(f"Failed to save prediction: {e}")
            
            return {
                'prediction': 'UP' if prediction_class == 1 else 'DOWN',
                'up_probability': float(up_probability),
                'down_probability': float(down_probability),
                'confidence': float(confidence),
                'signal_strength': float(signal_strength),
                'model_accuracy': float(model_data.get('accuracy', model_data.get('test_metrics', {}).get('accuracy', 0.0))),
                'model_cv_accuracy': float(model_data.get('cv_accuracy', model_data.get('cv_score', 0.0))),
                'missing_features_count': len(missing_features),
                'model_available': True,
                'high_confidence': confidence >= 0.75
            }
            
        except Exception as e:
            logger.error(f"Error predicting {symbol} {target_pct}: {e}")
            return {
                'error': f'Prediction failed: {str(e)}',
                'model_available': False
            }
            
    def predict_all_assets(self, symbols: List[str] = None) -> Dict[str, Dict]:
        """Make predictions for all assets and all targets.
        
        Args:
            symbols: List of symbols to predict. If None, uses config assets.
            
        Returns:
            Dictionary with predictions for all symbols
        """
        if symbols is None:
            symbols = config.assets
            
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
        """Create a summary DataFrame of all predictions.
        
        Args:
            predictions: Dictionary of predictions
            
        Returns:
            DataFrame with prediction summary
        """
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
                        'down_probability': pred_data['down_probability'],
                        'signal_strength': pred_data['signal_strength'],
                        'model_accuracy': pred_data.get('model_accuracy', 0),
                        'high_confidence': pred_data.get('high_confidence', False),
                        'timestamp': symbol_data['timestamp']
                    })
        
        if summary_data:
            return pd.DataFrame(summary_data)
        else:
            return pd.DataFrame()
            
    def get_high_confidence_alerts(self, predictions: Dict[str, Dict]) -> List[Dict]:
        """Extract high-confidence predictions for alerts.
        
        Args:
            predictions: Dictionary of predictions
            
        Returns:
            List of high-confidence alerts
        """
        alerts = []
        
        for symbol, symbol_data in predictions.items():
            if 'error' in symbol_data:
                continue
                
            for target_str, pred_data in symbol_data['predictions'].items():
                if 'error' not in pred_data and pred_data.get('high_confidence', False):
                    alerts.append({
                        'symbol': symbol,
                        'target': target_str,
                        'prediction': pred_data['prediction'],
                        'confidence': pred_data['confidence'],
                        'up_probability': pred_data['up_probability'],
                        'down_probability': pred_data['down_probability'],
                        'timestamp': symbol_data['timestamp']
                    })
        
        return alerts
        
    def load_all_models(self, symbols: List[str] = None) -> Dict[str, Dict[float, bool]]:
        """Load all trained models into memory.
        
        Args:
            symbols: List of symbols to load. If None, uses config assets.
            
        Returns:
            Dictionary mapping symbols to load results
        """
        if symbols is None:
            symbols = config.assets
            
        load_results = {}
        
        for symbol in symbols:
            symbol_results = {}
            for target_pct in self.target_percentages:
                model_data = self.trainer.load_model_from_disk(symbol, target_pct)
                if model_data:
                    self.models[(symbol, target_pct)] = model_data
                    symbol_results[target_pct] = True
                else:
                    symbol_results[target_pct] = False
            load_results[symbol] = symbol_results
            
        return load_results
        
    def validate_prediction_inputs(self, symbol: str) -> Dict[str, any]:
        """Validate that we can make predictions for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            'symbol': symbol,
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check data availability
        df = db_ops.get_latest_candles(symbol, '1h', limit=200, include_futures=True)
        if df.empty or len(df) < 50:
            validation['valid'] = False
            validation['errors'].append(f'Insufficient data: {len(df)} candles')
            return validation
            
        # Check model availability
        missing_models = []
        for target_pct in self.target_percentages:
            model_data = self.trainer.load_model_from_disk(symbol, target_pct)
            if model_data is None:
                missing_models.append(f"{target_pct:.1%}")
                
        if missing_models:
            validation['valid'] = False
            validation['errors'].append(f'Missing models: {", ".join(missing_models)}')
            
        # Check data freshness
        latest_timestamp = df['timestamp'].iloc[-1]
        hours_old = (datetime.now() - latest_timestamp).total_seconds() / 3600
        
        if hours_old > 2:
            validation['warnings'].append(f'Data is {hours_old:.1f} hours old')
            
        return validation
        
    def get_model_status(self, symbols: List[str] = None) -> Dict[str, Dict]:
        """Get status of all models for given symbols.
        
        Args:
            symbols: List of symbols to check
            
        Returns:
            Dictionary with model status information
        """
        if symbols is None:
            symbols = config.assets
            
        status = {}
        
        for symbol in symbols:
            symbol_status = {
                'symbol': symbol,
                'models_available': {},
                'data_status': {},
                'overall_ready': True
            }
            
            # Check data
            df = db_ops.get_latest_candles(symbol, '1h', limit=200, include_futures=True)
            symbol_status['data_status'] = {
                'candles': len(df),
                'latest_timestamp': str(df['timestamp'].iloc[-1]) if not df.empty else None,
                'hours_old': (datetime.now() - df['timestamp'].iloc[-1]).total_seconds() / 3600 if not df.empty else None
            }
            
            # Check models
            for target_pct in self.target_percentages:
                model_data = self.trainer.load_model_from_disk(symbol, target_pct)
                symbol_status['models_available'][f"{target_pct:.1%}"] = model_data is not None
                
            # Overall readiness
            symbol_status['overall_ready'] = (
                len(df) >= 50 and 
                all(symbol_status['models_available'].values())
            )
            
            status[symbol] = symbol_status
            
        return status
