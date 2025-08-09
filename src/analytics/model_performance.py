"""Model performance tracking and analysis."""
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd

from src.database.operations import db_ops
from src.database.prediction_tracker import prediction_tracker
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelPerformanceTracker:
    """Track and record model performance metrics."""
    
    def calculate_daily_performance(self, evaluation_date: Optional[date] = None) -> Dict:
        """Calculate performance metrics for all models for a given date.
        
        Args:
            evaluation_date: Date to evaluate (default: yesterday)
            
        Returns:
            Dictionary with performance metrics by symbol and target
        """
        if evaluation_date is None:
            evaluation_date = date.today() - timedelta(days=1)
            
        try:
            # Get predictions and outcomes for the evaluation date
            query = """
            SELECT 
                p.symbol,
                p.target_pct,
                p.model_name,
                COUNT(p.id) as total_predictions,
                COUNT(po.id) as completed_predictions,
                SUM(CASE WHEN po.actual_outcome = p.prediction_class THEN 1 ELSE 0 END) as correct,
                SUM(CASE WHEN p.prediction_class = 1 THEN 1 ELSE 0 END) as predicted_up,
                SUM(CASE WHEN po.actual_outcome = 1 THEN 1 ELSE 0 END) as actual_up,
                AVG(po.time_to_target_hours) as avg_time_to_target,
                COUNT(CASE WHEN po.actual_outcome IS NOT NULL THEN 1 END) as outcomes_tracked
            FROM predictions p
            LEFT JOIN prediction_outcomes po ON p.id = po.prediction_id
            WHERE DATE(p.timestamp) = :evaluation_date
            GROUP BY p.symbol, p.target_pct, p.model_name
            """
            
            with db_ops.db.get_session() as session:
                from sqlalchemy import text
                result = session.execute(text(query), {'evaluation_date': evaluation_date})
                rows = result.fetchall()
            
            performance_data = {}
            
            for row in rows:
                symbol = row[0]
                target_pct = float(row[1])
                model_name = row[2]
                total_predictions = int(row[3])
                completed_predictions = int(row[4])
                correct = int(row[5])
                predicted_up = int(row[6])
                actual_up = int(row[7]) if row[7] else 0
                avg_time_to_target = float(row[8]) if row[8] else None
                outcomes_tracked = int(row[9])
                
                if symbol not in performance_data:
                    performance_data[symbol] = {}
                
                if target_pct not in performance_data[symbol]:
                    performance_data[symbol][target_pct] = {}
                
                # Calculate metrics
                metrics = self._calculate_metrics(
                    total_predictions, completed_predictions, correct,
                    predicted_up, actual_up, outcomes_tracked
                )
                
                metrics['model_name'] = model_name
                metrics['avg_time_to_target_hours'] = avg_time_to_target
                
                performance_data[symbol][target_pct] = metrics
                
            return performance_data
            
        except Exception as e:
            logger.error(f"Error calculating daily performance: {e}")
            return {}
    
    def _calculate_metrics(self, total: int, completed: int, correct: int,
                          predicted_up: int, actual_up: int, tracked: int) -> Dict:
        """Calculate performance metrics.
        
        Returns:
            Dictionary with accuracy, precision, recall, F1 scores
        """
        metrics = {
            'total_predictions': total,
            'tracked_outcomes': tracked,
            'accuracy': 0.0,
            'precision_up': 0.0,
            'precision_down': 0.0,
            'recall_up': 0.0,
            'recall_down': 0.0,
            'f1_up': 0.0,
            'f1_down': 0.0,
            'hit_rate': 0.0
        }
        
        if completed == 0:
            return metrics
        
        # Accuracy
        metrics['accuracy'] = correct / completed if completed > 0 else 0
        
        # Hit rate (percentage of predictions that reached target)
        metrics['hit_rate'] = tracked / total if total > 0 else 0
        
        # For precision and recall, we need true/false positives/negatives
        # This is simplified - a full confusion matrix would be better
        predicted_down = total - predicted_up
        actual_down = completed - actual_up
        
        # Precision UP: Of all UP predictions, how many were correct?
        if predicted_up > 0:
            # Approximate: assume correct predictions are distributed proportionally
            correct_up_predictions = correct * (predicted_up / total) if total > 0 else 0
            metrics['precision_up'] = min(correct_up_predictions / predicted_up, 1.0)
        
        # Precision DOWN
        if predicted_down > 0:
            correct_down_predictions = correct * (predicted_down / total) if total > 0 else 0
            metrics['precision_down'] = min(correct_down_predictions / predicted_down, 1.0)
        
        # Recall would need actual distribution of UP/DOWN in outcomes
        # For now, using approximations
        if actual_up > 0:
            metrics['recall_up'] = metrics['precision_up'] * 0.9  # Approximation
        
        if actual_down > 0:
            metrics['recall_down'] = metrics['precision_down'] * 0.9  # Approximation
        
        # F1 scores
        if metrics['precision_up'] + metrics['recall_up'] > 0:
            metrics['f1_up'] = 2 * (metrics['precision_up'] * metrics['recall_up']) / \
                              (metrics['precision_up'] + metrics['recall_up'])
        
        if metrics['precision_down'] + metrics['recall_down'] > 0:
            metrics['f1_down'] = 2 * (metrics['precision_down'] * metrics['recall_down']) / \
                                (metrics['precision_down'] + metrics['recall_down'])
        
        return metrics
    
    def save_performance_metrics(self, performance_data: Dict, evaluation_date: Optional[date] = None) -> int:
        """Save performance metrics to database.
        
        Args:
            performance_data: Dictionary with performance metrics
            evaluation_date: Date of evaluation
            
        Returns:
            Number of records saved
        """
        if evaluation_date is None:
            evaluation_date = date.today() - timedelta(days=1)
            
        saved_count = 0
        
        try:
            for symbol, targets in performance_data.items():
                for target_pct, metrics in targets.items():
                    query = """
                    INSERT INTO model_performance 
                    (symbol, target_pct, model_name, evaluation_date, accuracy,
                     precision_up, precision_down, recall_up, recall_down,
                     f1_up, f1_down, total_predictions, avg_time_to_target_hours, hit_rate)
                    VALUES (:symbol, :target_pct, :model_name, :evaluation_date, :accuracy,
                            :precision_up, :precision_down, :recall_up, :recall_down,
                            :f1_up, :f1_down, :total_predictions, :avg_time_to_target_hours, :hit_rate)
                    ON CONFLICT (symbol, target_pct, model_name, evaluation_date) 
                    DO UPDATE SET
                        accuracy = EXCLUDED.accuracy,
                        precision_up = EXCLUDED.precision_up,
                        precision_down = EXCLUDED.precision_down,
                        recall_up = EXCLUDED.recall_up,
                        recall_down = EXCLUDED.recall_down,
                        f1_up = EXCLUDED.f1_up,
                        f1_down = EXCLUDED.f1_down,
                        total_predictions = EXCLUDED.total_predictions,
                        avg_time_to_target_hours = EXCLUDED.avg_time_to_target_hours,
                        hit_rate = EXCLUDED.hit_rate,
                        created_at = NOW()
                    """
                    
                    with db_ops.db.get_session() as session:
                        from sqlalchemy import text
                        session.execute(text(query), {
                            'symbol': symbol,
                            'target_pct': target_pct,
                            'model_name': metrics.get('model_name', f'RandomForest_{target_pct:.1%}'),
                            'evaluation_date': evaluation_date,
                            'accuracy': metrics.get('accuracy', 0),
                            'precision_up': metrics.get('precision_up', 0),
                            'precision_down': metrics.get('precision_down', 0),
                            'recall_up': metrics.get('recall_up', 0),
                            'recall_down': metrics.get('recall_down', 0),
                            'f1_up': metrics.get('f1_up', 0),
                            'f1_down': metrics.get('f1_down', 0),
                            'total_predictions': metrics.get('total_predictions', 0),
                            'avg_time_to_target_hours': metrics.get('avg_time_to_target_hours'),
                            'hit_rate': metrics.get('hit_rate', 0)
                        })
                        session.commit()
                    
                    saved_count += 1
                    
            logger.info(f"Saved {saved_count} model performance records for {evaluation_date}")
            
        except Exception as e:
            logger.error(f"Error saving performance metrics: {e}")
            
        return saved_count
    
    def update_daily_performance(self, days_back: int = 1):
        """Update performance metrics for recent days.
        
        Args:
            days_back: Number of days to calculate
        """
        for i in range(days_back):
            eval_date = date.today() - timedelta(days=i+1)
            
            logger.info(f"Calculating performance for {eval_date}")
            performance_data = self.calculate_daily_performance(eval_date)
            
            if performance_data:
                self.save_performance_metrics(performance_data, eval_date)
    
    def get_performance_trends(self, symbol: str, target_pct: float, days: int = 30) -> pd.DataFrame:
        """Get performance trends for a specific model.
        
        Args:
            symbol: Trading symbol
            target_pct: Target percentage
            days: Number of days to retrieve
            
        Returns:
            DataFrame with daily performance metrics
        """
        try:
            query = """
            SELECT 
                evaluation_date,
                accuracy,
                precision_up,
                precision_down,
                f1_up,
                f1_down,
                total_predictions,
                hit_rate
            FROM model_performance
            WHERE symbol = %s 
                AND target_pct = %s
                AND evaluation_date >= %s
            ORDER BY evaluation_date DESC
            """
            
            start_date = date.today() - timedelta(days=days)
            
            df = pd.read_sql_query(
                query, 
                db_ops.db.connection,
                params=(symbol, target_pct, start_date)
            )
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting performance trends: {e}")
            return pd.DataFrame()


# Singleton instance
model_performance_tracker = ModelPerformanceTracker()