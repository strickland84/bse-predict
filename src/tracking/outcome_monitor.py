"""Monitor and track prediction outcomes."""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from src.database.operations import db_ops
from src.database.prediction_tracker import prediction_tracker
from src.utils.logger import get_logger
from src.utils.config import config

logger = get_logger(__name__)


class OutcomeMonitor:
    """Monitors predictions and tracks their outcomes."""
    
    def __init__(self, monitoring_window_hours: int = 72):
        """Initialize outcome monitor.
        
        Args:
            monitoring_window_hours: How long to monitor each prediction
        """
        self.monitoring_window_hours = monitoring_window_hours
        self.db_ops = db_ops
        
    def check_prediction_outcomes(self) -> Dict[str, int]:
        """Check outcomes for all pending predictions.
        
        Returns:
            Dictionary with outcome statistics
        """
        logger.info("Starting prediction outcome check...")
        
        # Get pending predictions
        pending_df = prediction_tracker.get_pending_predictions(
            window_hours=self.monitoring_window_hours
        )
        
        if pending_df.empty:
            logger.info("No pending predictions to check")
            return {'checked': 0, 'completed': 0, 'hit_target': 0, 'missed_target': 0}
        
        stats = {
            'checked': len(pending_df),
            'completed': 0,
            'hit_target': 0,
            'missed_target': 0,
            'expired': 0,
            'still_pending': 0,
            'completed_details': []  # Track details of completed predictions
        }
        
        # Process each pending prediction
        for _, prediction in pending_df.iterrows():
            outcome = self._check_single_prediction(prediction)
            
            if outcome['completed']:
                stats['completed'] += 1
                
                # Add completed prediction details
                detail = {
                    'symbol': prediction['symbol'],
                    'target_pct': prediction['target_pct'],
                    'predicted': 'UP' if prediction['prediction_class'] == 1 else 'DOWN',
                    'result': None,
                    'time_to_target': outcome.get('time_to_target_hours')
                }
                
                if outcome.get('first_target_hit'):
                    if outcome.get('prediction_correct'):
                        stats['hit_target'] += 1
                        detail['result'] = 'CORRECT'
                    else:
                        stats['missed_target'] += 1
                        detail['result'] = 'INCORRECT'
                else:
                    # No target hit - expired
                    stats['expired'] += 1
                    detail['result'] = 'EXPIRED'
                
                stats['completed_details'].append(detail)
            else:
                stats['still_pending'] += 1
                
        logger.info(f"Outcome check complete: {stats}")
        return stats
    
    def _check_single_prediction(self, prediction: pd.Series) -> Dict:
        """Check outcome for a single prediction.
        
        This checks which target (UP or DOWN) is hit FIRST, matching how the models were trained.
        The prediction is correct if the predicted direction's target is hit before the opposite target.
        
        Args:
            prediction: Series with prediction data
            
        Returns:
            Dictionary with outcome information
        """
        try:
            # Get price data since prediction
            symbol = prediction['symbol']
            pred_timestamp = prediction['timestamp']
            target_pct = float(prediction['target_pct'])
            prediction_class = int(prediction['prediction_class'])  # 1 for UP, 0 for DOWN
            
            # Get OHLCV data starting from prediction timestamp
            # Get enough candles to cover the monitoring window plus buffer
            candles_needed = self.monitoring_window_hours + 2  # 72 hours + 2 for buffer
            df = self.db_ops.get_candles_from_timestamp(
                symbol, pred_timestamp, '1h', limit=candles_needed, include_futures=False
            )
            
            if df.empty or len(df) < 2:
                logger.debug(f"Insufficient data for prediction {prediction['id']}")
                return {'completed': False, 'hit_target': False}
            
            # Get the prediction candle price (close price at prediction time)
            prediction_price = df.iloc[0]['close']
            
            # Calculate BOTH target prices (we need to check which is hit first)
            up_target_price = prediction_price * (1 + target_pct)
            down_target_price = prediction_price * (1 - target_pct)
            
            # Check if monitoring window has expired
            monitoring_end = pred_timestamp + timedelta(hours=self.monitoring_window_hours)
            current_time = datetime.now(timezone.utc)
            
            if current_time > monitoring_end:
                # Monitoring window expired, mark as completed
                monitoring_df = df[df['timestamp'] <= monitoring_end]
            else:
                # Still within monitoring window
                monitoring_df = df
                
            # Skip the first candle (prediction candle) for outcome checking
            if len(monitoring_df) > 1:
                outcome_df = monitoring_df.iloc[1:]
            else:
                # Not enough data yet
                return {'completed': False, 'hit_target': False}
            
            # Track price movements for analytics
            max_price = outcome_df['high'].max()
            min_price = outcome_df['low'].min()
            
            # Calculate max moves from prediction price
            max_up_move = (max_price / prediction_price - 1) * 100
            max_down_move = (1 - min_price / prediction_price) * 100
            
            # Check which target is hit FIRST
            first_target_hit = None  # Will be 'UP', 'DOWN', or None
            target_hit_timestamp = None
            time_to_target_hours = None
            
            # Iterate through each candle to find which target is hit first
            for idx, candle in outcome_df.iterrows():
                up_hit = candle['high'] >= up_target_price
                down_hit = candle['low'] <= down_target_price
                
                if up_hit and down_hit:
                    # Both targets hit in same candle (rare but possible in volatile markets)
                    # Determine which was likely hit first based on open price
                    open_price = candle['open']
                    up_distance = abs(open_price - up_target_price)
                    down_distance = abs(open_price - down_target_price)
                    
                    if up_distance < down_distance:
                        first_target_hit = 'UP'
                    else:
                        first_target_hit = 'DOWN'
                    
                    target_hit_timestamp = candle['timestamp']
                    logger.info(f"Both targets hit in same candle for {symbol} {target_pct:.1%}. "
                              f"Determined {first_target_hit} was hit first based on open price.")
                    break
                    
                elif up_hit:
                    first_target_hit = 'UP'
                    target_hit_timestamp = candle['timestamp']
                    break
                    
                elif down_hit:
                    first_target_hit = 'DOWN'
                    target_hit_timestamp = candle['timestamp']
                    break
            
            # Calculate time to target if hit
            if target_hit_timestamp:
                time_to_target_hours = (target_hit_timestamp - pred_timestamp).total_seconds() / 3600
            
            # Determine if prediction was correct
            prediction_correct = False
            if first_target_hit:
                if prediction_class == 1 and first_target_hit == 'UP':
                    prediction_correct = True
                elif prediction_class == 0 and first_target_hit == 'DOWN':
                    prediction_correct = True
            
            # Set favorable and adverse moves based on prediction direction
            if prediction_class == 1:  # UP prediction
                max_favorable_move = max_up_move
                max_adverse_move = max_down_move
            else:  # DOWN prediction
                max_favorable_move = max_down_move
                max_adverse_move = max_up_move
            
            # Determine if monitoring is complete
            completed = current_time > monitoring_end or first_target_hit is not None
            
            if completed:
                # Update database with outcome
                # actual_outcome represents what actually happened (1 for UP, 0 for DOWN, -1 for expired)
                if first_target_hit:
                    actual_outcome = 1 if first_target_hit == 'UP' else 0
                else:
                    # No target hit within window - mark as expired/no result
                    # Using -1 to indicate no target was hit (neither UP nor DOWN)
                    actual_outcome = -1
                    logger.info(f"Prediction {prediction['id']} expired without hitting either target")
                    
                success = prediction_tracker.update_prediction_outcome(
                    prediction_id=int(prediction['id']),
                    actual_outcome=int(actual_outcome) if actual_outcome is not None else None,
                    target_hit_timestamp=target_hit_timestamp,
                    time_to_target_hours=float(time_to_target_hours) if time_to_target_hours is not None else None,
                    max_favorable_move=float(max_favorable_move) if max_favorable_move is not None else None,
                    max_adverse_move=float(max_adverse_move) if max_adverse_move is not None else None
                )
                
                if success:
                    result_str = "CORRECT" if prediction_correct else "INCORRECT"
                    if first_target_hit:
                        logger.info(
                            f"Updated outcome for {symbol} {target_pct:.1%} prediction: "
                            f"{result_str} (Predicted {'UP' if prediction_class == 1 else 'DOWN'}, "
                            f"Hit {first_target_hit} first after {time_to_target_hours:.1f}h)"
                        )
                    else:
                        logger.info(
                            f"Updated outcome for {symbol} {target_pct:.1%} prediction: "
                            f"EXPIRED (No target hit within {self.monitoring_window_hours}h window)"
                        )
                    
            return {
                'completed': completed,
                'hit_target': first_target_hit is not None,
                'first_target_hit': first_target_hit,
                'prediction_correct': prediction_correct,
                'target_hit_timestamp': target_hit_timestamp,
                'time_to_target_hours': time_to_target_hours,
                'max_favorable_move': max_favorable_move,
                'max_adverse_move': max_adverse_move
            }
            
        except Exception as e:
            logger.error(f"Error checking prediction {prediction['id']}: {e}")
            return {'completed': False, 'hit_target': False}
    
    def get_outcome_summary(self, hours_back: int = 24) -> pd.DataFrame:
        """Get summary of recent prediction outcomes.
        
        Args:
            hours_back: Hours to look back
            
        Returns:
            DataFrame with outcome summary
        """
        recent_preds = prediction_tracker.get_recent_predictions(hours_back)
        
        if recent_preds.empty:
            return pd.DataFrame()
        
        # Calculate summary statistics
        summary_data = []
        
        for symbol in recent_preds['symbol'].unique():
            symbol_data = recent_preds[recent_preds['symbol'] == symbol]
            
            for target_pct in symbol_data['target_pct'].unique():
                target_data = symbol_data[symbol_data['target_pct'] == target_pct]
                
                completed = target_data[target_data['status'] != 'Pending']
                # For accuracy, only count predictions that hit a target (exclude expired)
                hit_target = completed[completed['status'].isin(['Correct', 'Incorrect'])]
                if len(hit_target) > 0:
                    accuracy = len(hit_target[hit_target['status'] == 'Correct']) / len(hit_target)
                else:
                    accuracy = 0
                    
                summary_data.append({
                    'symbol': symbol,
                    'target_pct': target_pct,
                    'total_predictions': len(target_data),
                    'completed': len(completed),
                    'pending': len(target_data[target_data['status'] == 'Pending']),
                    'expired': len(completed[completed['status'] == 'Expired']),
                    'correct': len(hit_target[hit_target['status'] == 'Correct']) if len(hit_target) > 0 else 0,
                    'incorrect': len(hit_target[hit_target['status'] == 'Incorrect']) if len(hit_target) > 0 else 0,
                    'accuracy': accuracy,
                    'avg_confidence': target_data['confidence'].mean(),
                    'avg_time_to_target': hit_target['time_to_target_hours'].mean() if len(hit_target) > 0 else None
                })
                
        return pd.DataFrame(summary_data)
    
    def get_detailed_outcomes(self, symbol: Optional[str] = None,
                            target_pct: Optional[float] = None,
                            days_back: int = 7) -> pd.DataFrame:
        """Get detailed prediction outcomes with all metrics.
        
        Args:
            symbol: Filter by symbol
            target_pct: Filter by target percentage
            days_back: Days to look back
            
        Returns:
            DataFrame with detailed outcomes
        """
        return prediction_tracker.get_prediction_performance(
            symbol, target_pct, days_back
        )
    
    def generate_performance_report(self) -> str:
        """Generate a text report of prediction performance.
        
        Returns:
            Formatted performance report
        """
        # Get 24-hour summary
        summary = self.get_outcome_summary(24)
        
        if summary.empty:
            return "No prediction data available for the last 24 hours."
        
        report_lines = [
            "📊 PREDICTION PERFORMANCE REPORT (24h)",
            "=" * 40
        ]
        
        # Overall statistics
        total_predictions = summary['total_predictions'].sum()
        total_completed = summary['completed'].sum()
        total_correct = summary['correct'].sum()
        
        if total_completed > 0:
            overall_accuracy = total_correct / total_completed
            report_lines.extend([
                f"Total Predictions: {total_predictions}",
                f"Completed: {total_completed}",
                f"Overall Accuracy: {overall_accuracy:.1%}",
                ""
            ])
        
        # Per symbol/target breakdown
        report_lines.append("BREAKDOWN BY SYMBOL & TARGET:")
        report_lines.append("-" * 30)
        
        for _, row in summary.iterrows():
            symbol = row['symbol']
            target = row['target_pct']
            accuracy = row['accuracy']
            completed = row['completed']
            pending = row['pending']
            
            report_lines.append(
                f"{symbol} ±{target:.1%}: "
                f"{accuracy:.1%} accuracy "
                f"({completed} completed, {pending} pending)"
            )
        
        # Get 7-day performance for context
        week_performance = prediction_tracker.get_prediction_performance(days_back=7)
        
        if not week_performance.empty:
            report_lines.extend([
                "",
                "7-DAY PERFORMANCE TRENDS:",
                "-" * 30
            ])
            
            for _, row in week_performance.iterrows():
                if row['tracked_predictions'] > 0:
                    report_lines.append(
                        f"{row['symbol']} ±{row['target_pct']:.1%}: "
                        f"{row['accuracy']:.1%} accuracy "
                        f"({row['tracked_predictions']} predictions)"
                    )
        
        return "\n".join(report_lines)


# Create singleton instance
outcome_monitor = OutcomeMonitor(monitoring_window_hours=72)