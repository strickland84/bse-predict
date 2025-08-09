"""Database operations for prediction tracking and outcome monitoring."""
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
from decimal import Decimal

from src.database.connection import DatabaseConnection
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PredictionTracker:
    """Handles database operations for predictions and outcomes."""
    
    def __init__(self):
        from src.database.operations import db_connection
        self.conn = db_connection
        
    def save_prediction(self, symbol: str, target_pct: float, timestamp: datetime,
                       model_name: str, prediction_class: int, probability: float,
                       confidence: float, features_used: Dict) -> Optional[int]:
        """Save a prediction to the database.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            target_pct: Target percentage (0.01, 0.02, 0.05)
            timestamp: Prediction timestamp
            model_name: Name of the model used
            prediction_class: 0 for DOWN, 1 for UP
            probability: Probability of the prediction
            confidence: Confidence score
            features_used: Dictionary of features used
            
        Returns:
            Prediction ID if successful, None otherwise
        """
        query = """
            INSERT INTO predictions 
            (symbol, target_pct, timestamp, model_name, prediction_class, 
             probability, confidence, features_used)
            VALUES (:symbol, :target_pct, :timestamp, :model_name, :prediction_class, 
                    :probability, :confidence, :features_used)
            ON CONFLICT (symbol, target_pct, timestamp, model_name) 
            DO UPDATE SET
                prediction_class = EXCLUDED.prediction_class,
                probability = EXCLUDED.probability,
                confidence = EXCLUDED.confidence,
                features_used = EXCLUDED.features_used,
                created_at = NOW()
            RETURNING id
        """
        
        try:
            with self.conn.get_session() as session:
                from sqlalchemy import text
                result = session.execute(
                    text(query), 
                    {
                        'symbol': symbol, 
                        'target_pct': target_pct, 
                        'timestamp': timestamp, 
                        'model_name': model_name, 
                        'prediction_class': prediction_class,
                        'probability': probability, 
                        'confidence': confidence, 
                        'features_used': json.dumps(features_used)
                    }
                )
                session.commit()
                
                prediction_id = result.fetchone()
                if prediction_id:
                    prediction_id = prediction_id[0]
                    logger.info(f"Saved prediction {prediction_id} for {symbol} {target_pct:.1%}")
                    return prediction_id
                return None
            
        except Exception as e:
            logger.error(f"Error saving prediction: {e}")
            return None
    
    def save_predictions_batch(self, predictions: List[Dict]) -> List[int]:
        """Save multiple predictions in a batch.
        
        Args:
            predictions: List of prediction dictionaries
            
        Returns:
            List of prediction IDs
        """
        if not predictions:
            return []
            
        query = """
            INSERT INTO predictions 
            (symbol, target_pct, timestamp, model_name, prediction_class, 
             probability, confidence, features_used)
            VALUES (:symbol, :target_pct, :timestamp, :model_name, :prediction_class,
                    :probability, :confidence, :features_used)
            ON CONFLICT (symbol, target_pct, timestamp, model_name) 
            DO UPDATE SET
                prediction_class = EXCLUDED.prediction_class,
                probability = EXCLUDED.probability,
                confidence = EXCLUDED.confidence,
                features_used = EXCLUDED.features_used,
                created_at = NOW()
            RETURNING id
        """
        
        prediction_ids = []
        
        with self.conn.get_session() as session:
            from sqlalchemy import text
            for pred in predictions:
                try:
                    result = session.execute(
                        text(query),
                        {
                            'symbol': pred['symbol'], 
                            'target_pct': pred['target_pct'], 
                            'timestamp': pred['timestamp'], 
                            'model_name': pred['model_name'], 
                            'prediction_class': pred['prediction_class'],
                            'probability': pred['probability'], 
                            'confidence': pred['confidence'], 
                            'features_used': json.dumps(pred.get('features_used', {}))
                        }
                    )
                    
                    prediction_id = result.fetchone()
                    if prediction_id:
                        prediction_ids.append(prediction_id[0])
                        
                except Exception as e:
                    logger.error(f"Error saving prediction batch item: {e}")
            
            session.commit()
                
        logger.info(f"Saved {len(prediction_ids)}/{len(predictions)} predictions")
        return prediction_ids
    
    def get_pending_predictions(self, window_hours: int = 24) -> pd.DataFrame:
        """Get predictions that need outcome tracking.
        
        Args:
            window_hours: Hours to look back for predictions
            
        Returns:
            DataFrame with pending predictions
        """
        query = """
            SELECT 
                p.id, p.symbol, p.target_pct, p.timestamp, p.model_name,
                p.prediction_class, p.probability, p.confidence,
                p.features_used, p.created_at
            FROM predictions p
            LEFT JOIN prediction_outcomes po ON p.id = po.prediction_id
            WHERE po.id IS NULL
                AND p.timestamp >= NOW() - CAST(:window_hours || ' hours' AS INTERVAL)
                AND p.timestamp <= NOW() - INTERVAL '1 hour'
            ORDER BY p.timestamp DESC
        """
        
        try:
            with self.conn.get_session() as session:
                from sqlalchemy import text
                result = session.execute(text(query), {'window_hours': window_hours})
                rows = result.fetchall()
                
                if rows:
                    columns = ['id', 'symbol', 'target_pct', 'timestamp', 'model_name',
                              'prediction_class', 'probability', 'confidence',
                              'features_used', 'created_at']
                    df = pd.DataFrame(rows, columns=columns)
                    logger.info(f"Found {len(df)} pending predictions to track")
                    return df
                else:
                    logger.info("No pending predictions found")
                    return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error getting pending predictions: {e}")
            return pd.DataFrame()
    
    def update_prediction_outcome(self, prediction_id: int, actual_outcome: int,
                                target_hit_timestamp: Optional[datetime] = None,
                                time_to_target_hours: Optional[float] = None,
                                max_favorable_move: Optional[float] = None,
                                max_adverse_move: Optional[float] = None) -> bool:
        """Update the outcome of a prediction.
        
        Args:
            prediction_id: ID of the prediction
            actual_outcome: 0 for DOWN, 1 for UP, None if not hit
            target_hit_timestamp: When target was hit
            time_to_target_hours: Hours to reach target
            max_favorable_move: Maximum move in predicted direction
            max_adverse_move: Maximum move against prediction
            
        Returns:
            True if successful
        """
        query = """
            INSERT INTO prediction_outcomes
            (prediction_id, actual_outcome, target_hit_timestamp, 
             time_to_target_hours, max_favorable_move, max_adverse_move)
            VALUES (:prediction_id, :actual_outcome, :target_hit_timestamp, 
                    :time_to_target_hours, :max_favorable_move, :max_adverse_move)
            ON CONFLICT (prediction_id) DO UPDATE SET
                actual_outcome = EXCLUDED.actual_outcome,
                target_hit_timestamp = EXCLUDED.target_hit_timestamp,
                time_to_target_hours = EXCLUDED.time_to_target_hours,
                max_favorable_move = EXCLUDED.max_favorable_move,
                max_adverse_move = EXCLUDED.max_adverse_move,
                completed_at = NOW()
        """
        
        try:
            with self.conn.get_session() as session:
                from sqlalchemy import text
                session.execute(
                    text(query),
                    {
                        'prediction_id': prediction_id, 
                        'actual_outcome': actual_outcome, 
                        'target_hit_timestamp': target_hit_timestamp,
                        'time_to_target_hours': time_to_target_hours, 
                        'max_favorable_move': max_favorable_move, 
                        'max_adverse_move': max_adverse_move
                    }
                )
                session.commit()
            logger.info(f"Updated outcome for prediction {prediction_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating prediction outcome: {e}")
            return False
    
    def get_prediction_performance(self, symbol: Optional[str] = None,
                                 target_pct: Optional[float] = None,
                                 days_back: int = 7) -> pd.DataFrame:
        """Get prediction performance statistics.
        
        Args:
            symbol: Filter by symbol (optional)
            target_pct: Filter by target percentage (optional)
            days_back: Number of days to analyze
            
        Returns:
            DataFrame with performance metrics
        """
        where_clauses = ["p.timestamp >= NOW() - INTERVAL :days"]
        params = {'days': f'{days_back} days'}
        
        if symbol:
            where_clauses.append("p.symbol = :symbol")
            params['symbol'] = symbol
            
        if target_pct:
            where_clauses.append("p.target_pct = :target_pct")
            params['target_pct'] = target_pct
            
        where_clause = " AND ".join(where_clauses)
        
        query = f"""
            SELECT 
                p.symbol,
                p.target_pct,
                p.model_name,
                COUNT(p.id) as total_predictions,
                COUNT(po.id) as tracked_predictions,
                SUM(CASE WHEN po.actual_outcome = -1 THEN 1 ELSE 0 END) as expired_predictions,
                SUM(CASE WHEN po.actual_outcome = p.prediction_class THEN 1 ELSE 0 END) as correct_predictions,
                -- Only calculate accuracy on predictions that hit a target (exclude expired)
                AVG(CASE 
                    WHEN po.actual_outcome != -1 THEN 
                        CASE WHEN po.actual_outcome = p.prediction_class THEN 1.0 ELSE 0.0 END
                    ELSE NULL 
                END) as accuracy,
                AVG(p.confidence) as avg_confidence,
                AVG(po.time_to_target_hours) as avg_time_to_target,
                AVG(po.max_favorable_move) as avg_favorable_move,
                AVG(po.max_adverse_move) as avg_adverse_move
            FROM predictions p
            LEFT JOIN prediction_outcomes po ON p.id = po.prediction_id
            WHERE {where_clause}
            GROUP BY p.symbol, p.target_pct, p.model_name
            ORDER BY p.symbol, p.target_pct, accuracy DESC
        """
        
        try:
            with self.conn.get_session() as session:
                from sqlalchemy import text
                result = session.execute(text(query), params)
                rows = result.fetchall()
                
                if rows:
                    columns = ['symbol', 'target_pct', 'model_name', 'total_predictions',
                              'tracked_predictions', 'expired_predictions', 'correct_predictions', 
                              'accuracy', 'avg_confidence', 'avg_time_to_target', 
                              'avg_favorable_move', 'avg_adverse_move']
                    df = pd.DataFrame(rows, columns=columns)
                    return df
                else:
                    return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error getting prediction performance: {e}")
            return pd.DataFrame()
    
    def get_recent_predictions(self, hours: int = 24) -> pd.DataFrame:
        """Get recent predictions with their outcomes.
        
        Args:
            hours: Hours to look back
            
        Returns:
            DataFrame with recent predictions and outcomes
        """
        query = """
            SELECT 
                p.id, p.symbol, p.target_pct, p.timestamp, p.prediction_class,
                p.probability, p.confidence,
                po.actual_outcome, po.target_hit_timestamp, po.time_to_target_hours,
                po.max_favorable_move, po.max_adverse_move,
                CASE 
                    WHEN po.actual_outcome IS NULL THEN 'Pending'
                    WHEN po.actual_outcome = -1 THEN 'Expired'
                    WHEN po.actual_outcome = p.prediction_class THEN 'Correct'
                    ELSE 'Incorrect'
                END as status
            FROM predictions p
            LEFT JOIN prediction_outcomes po ON p.id = po.prediction_id
            WHERE p.timestamp >= NOW() - INTERVAL :hours
            ORDER BY p.timestamp DESC
        """
        
        try:
            with self.conn.get_session() as session:
                from sqlalchemy import text
                result = session.execute(text(query), {'hours': f'{hours} hours'})
                rows = result.fetchall()
                
                if rows:
                    columns = ['id', 'symbol', 'target_pct', 'timestamp', 'prediction_class',
                              'probability', 'confidence', 'actual_outcome', 'target_hit_timestamp',
                              'time_to_target_hours', 'max_favorable_move', 'max_adverse_move', 'status']
                    df = pd.DataFrame(rows, columns=columns)
                    return df
                else:
                    return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error getting recent predictions: {e}")
            return pd.DataFrame()
    
    def cleanup_old_predictions(self, days_to_keep: int = 30) -> int:
        """Clean up old predictions and outcomes.
        
        Args:
            days_to_keep: Number of days to retain
            
        Returns:
            Number of records deleted
        """
        query = """
            DELETE FROM predictions 
            WHERE timestamp < NOW() - INTERVAL :days
            RETURNING id
        """
        
        try:
            with self.conn.get_session() as session:
                from sqlalchemy import text
                result = session.execute(
                    text(query), 
                    {'days': f'{days_to_keep} days'}
                )
                deleted_ids = result.fetchall()
                session.commit()
                
                deleted_count = len(deleted_ids) if deleted_ids else 0
                logger.info(f"Cleaned up {deleted_count} old predictions")
                return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up predictions: {e}")
            return 0


# Singleton instance
prediction_tracker = PredictionTracker()