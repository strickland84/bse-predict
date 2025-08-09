"""Analytics for prediction performance and insights."""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json

from src.database.prediction_tracker import prediction_tracker
from src.utils.logger import get_logger
from src.utils.config import config

logger = get_logger(__name__)


class PredictionAnalytics:
    """Analyze prediction performance and generate insights."""
    
    def __init__(self):
        self.tracker = prediction_tracker
        
    def get_overall_performance(self, days_back: int = 30) -> Dict:
        """Get overall system performance metrics.
        
        Args:
            days_back: Number of days to analyze
            
        Returns:
            Dictionary with performance metrics
        """
        try:
            # Get performance data
            perf_df = self.tracker.get_prediction_performance(days_back=days_back)
            
            if perf_df.empty:
                return {
                    'total_predictions': 0,
                    'overall_accuracy': 0,
                    'best_performing': None,
                    'worst_performing': None
                }
            
            # Calculate overall metrics
            total_predictions = perf_df['total_predictions'].sum()
            tracked_predictions = perf_df['tracked_predictions'].sum()
            correct_predictions = perf_df['correct_predictions'].sum()
            
            overall_accuracy = correct_predictions / tracked_predictions if tracked_predictions > 0 else 0
            
            # Find best and worst performing configurations
            perf_df_sorted = perf_df.sort_values('accuracy', ascending=False)
            best_config = perf_df_sorted.iloc[0] if not perf_df_sorted.empty else None
            worst_config = perf_df_sorted.iloc[-1] if not perf_df_sorted.empty else None
            
            return {
                'total_predictions': int(total_predictions),
                'tracked_predictions': int(tracked_predictions),
                'overall_accuracy': float(overall_accuracy),
                'avg_confidence': float(perf_df['avg_confidence'].mean()),
                'avg_time_to_target': float(perf_df['avg_time_to_target'].mean()),
                'best_performing': {
                    'symbol': best_config['symbol'],
                    'target_pct': float(best_config['target_pct']),
                    'accuracy': float(best_config['accuracy'])
                } if best_config is not None else None,
                'worst_performing': {
                    'symbol': worst_config['symbol'],
                    'target_pct': float(worst_config['target_pct']),
                    'accuracy': float(worst_config['accuracy'])
                } if worst_config is not None else None
            }
            
        except Exception as e:
            logger.error(f"Error calculating overall performance: {e}")
            return {}
    
    def get_symbol_performance(self, symbol: str, days_back: int = 30) -> Dict:
        """Get performance metrics for a specific symbol.
        
        Args:
            symbol: Trading pair symbol
            days_back: Number of days to analyze
            
        Returns:
            Dictionary with symbol performance metrics
        """
        try:
            perf_df = self.tracker.get_prediction_performance(
                symbol=symbol, days_back=days_back
            )
            
            if perf_df.empty:
                return {'symbol': symbol, 'no_data': True}
            
            # Calculate metrics by target
            target_performance = {}
            
            for _, row in perf_df.iterrows():
                target_key = f"{row['target_pct']:.1%}"
                target_performance[target_key] = {
                    'accuracy': float(row['accuracy']) if pd.notna(row['accuracy']) else 0,
                    'total_predictions': int(row['total_predictions']),
                    'tracked_predictions': int(row['tracked_predictions']),
                    'avg_confidence': float(row['avg_confidence']) if pd.notna(row['avg_confidence']) else 0,
                    'avg_time_to_target': float(row['avg_time_to_target']) if pd.notna(row['avg_time_to_target']) else 0
                }
            
            # Overall symbol metrics
            total_predictions = perf_df['total_predictions'].sum()
            tracked_predictions = perf_df['tracked_predictions'].sum()
            correct_predictions = perf_df['correct_predictions'].sum()
            
            return {
                'symbol': symbol,
                'total_predictions': int(total_predictions),
                'tracked_predictions': int(tracked_predictions),
                'overall_accuracy': float(correct_predictions / tracked_predictions) if tracked_predictions > 0 else 0,
                'by_target': target_performance
            }
            
        except Exception as e:
            logger.error(f"Error getting symbol performance: {e}")
            return {'symbol': symbol, 'error': str(e)}
    
    def get_accuracy_trends(self, symbol: Optional[str] = None,
                           target_pct: Optional[float] = None,
                           days: int = 7) -> pd.DataFrame:
        """Get daily accuracy trends.
        
        Args:
            symbol: Filter by symbol
            target_pct: Filter by target
            days: Number of days to analyze
            
        Returns:
            DataFrame with daily accuracy trends
        """
        try:
            # Get raw prediction data
            end_date = datetime.now()
            daily_accuracy = []
            
            for i in range(days):
                date = end_date - timedelta(days=i)
                
                # Get predictions for this day
                day_perf = self.tracker.get_prediction_performance(
                    symbol=symbol,
                    target_pct=target_pct,
                    days_back=i+1
                )
                
                if not day_perf.empty:
                    # Filter to just this day's data
                    total_tracked = day_perf['tracked_predictions'].sum()
                    total_correct = day_perf['correct_predictions'].sum()
                    
                    if total_tracked > 0:
                        daily_accuracy.append({
                            'date': date.date(),
                            'accuracy': total_correct / total_tracked,
                            'predictions': total_tracked
                        })
            
            return pd.DataFrame(daily_accuracy)
            
        except Exception as e:
            logger.error(f"Error getting accuracy trends: {e}")
            return pd.DataFrame()
    
    def get_confidence_calibration(self) -> Dict:
        """Analyze how well confidence scores match actual accuracy.
        
        Returns:
            Dictionary with calibration metrics
        """
        try:
            # Get recent predictions with outcomes
            recent_preds = self.tracker.get_recent_predictions(hours=24*7)
            
            if recent_preds.empty:
                return {'calibrated': False, 'message': 'No data available'}
            
            # Group by confidence buckets
            completed = recent_preds[recent_preds['status'] != 'Pending']
            
            if completed.empty:
                return {'calibrated': False, 'message': 'No completed predictions'}
            
            # Create confidence buckets
            completed['confidence_bucket'] = pd.cut(
                completed['confidence'], 
                bins=[0, 0.6, 0.7, 0.8, 0.9, 1.0],
                labels=['50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
            )
            
            calibration_data = []
            
            for bucket in completed['confidence_bucket'].unique():
                if pd.notna(bucket):
                    bucket_data = completed[completed['confidence_bucket'] == bucket]
                    if len(bucket_data) > 0:
                        accuracy = len(bucket_data[bucket_data['status'] == 'Correct']) / len(bucket_data)
                        avg_confidence = bucket_data['confidence'].mean()
                        
                        calibration_data.append({
                            'bucket': str(bucket),
                            'avg_confidence': float(avg_confidence),
                            'actual_accuracy': float(accuracy),
                            'count': len(bucket_data),
                            'calibration_error': abs(avg_confidence - accuracy)
                        })
            
            # Calculate overall calibration score
            if calibration_data:
                avg_calibration_error = np.mean([d['calibration_error'] for d in calibration_data])
                is_calibrated = avg_calibration_error < 0.1  # Within 10% is considered calibrated
            else:
                avg_calibration_error = 1.0
                is_calibrated = False
            
            return {
                'calibrated': is_calibrated,
                'avg_calibration_error': float(avg_calibration_error),
                'buckets': calibration_data
            }
            
        except Exception as e:
            logger.error(f"Error calculating calibration: {e}")
            return {'calibrated': False, 'error': str(e)}
    
    def get_best_conditions(self, min_predictions: int = 10) -> List[Dict]:
        """Find conditions where predictions perform best.
        
        Args:
            min_predictions: Minimum predictions to consider
            
        Returns:
            List of best performing conditions
        """
        try:
            # Get performance data
            perf_df = self.tracker.get_prediction_performance(days_back=30)
            
            if perf_df.empty:
                return []
            
            # Filter by minimum predictions
            perf_df = perf_df[perf_df['tracked_predictions'] >= min_predictions]
            
            if perf_df.empty:
                return []
            
            # Sort by accuracy
            perf_df = perf_df.sort_values('accuracy', ascending=False)
            
            # Get top conditions
            best_conditions = []
            
            for _, row in perf_df.head(5).iterrows():
                best_conditions.append({
                    'symbol': row['symbol'],
                    'target_pct': float(row['target_pct']),
                    'accuracy': float(row['accuracy']),
                    'predictions': int(row['tracked_predictions']),
                    'avg_time_to_target': float(row['avg_time_to_target']) if pd.notna(row['avg_time_to_target']) else None
                })
            
            return best_conditions
            
        except Exception as e:
            logger.error(f"Error finding best conditions: {e}")
            return []
    
    def generate_analytics_report(self) -> str:
        """Generate comprehensive analytics report.
        
        Returns:
            Formatted analytics report
        """
        try:
            report_lines = [
                "📊 PREDICTION ANALYTICS REPORT",
                "=" * 40,
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}",
                ""
            ]
            
            # Overall performance
            overall = self.get_overall_performance(days_back=7)
            
            if overall.get('total_predictions', 0) > 0:
                report_lines.extend([
                    "📈 OVERALL PERFORMANCE (7 days):",
                    f"Total Predictions: {overall['total_predictions']}",
                    f"Tracked Outcomes: {overall['tracked_predictions']}",
                    f"Overall Accuracy: {overall['overall_accuracy']:.1%}",
                    f"Avg Confidence: {overall['avg_confidence']:.1%}",
                    ""
                ])
                
                if overall.get('best_performing'):
                    best = overall['best_performing']
                    report_lines.append(
                        f"🏆 Best: {best['symbol']} ±{best['target_pct']:.1%} "
                        f"({best['accuracy']:.1%} accuracy)"
                    )
                
                if overall.get('worst_performing'):
                    worst = overall['worst_performing']
                    report_lines.append(
                        f"📉 Worst: {worst['symbol']} ±{worst['target_pct']:.1%} "
                        f"({worst['accuracy']:.1%} accuracy)"
                    )
                
                report_lines.append("")
            
            # Confidence calibration
            calibration = self.get_confidence_calibration()
            
            if calibration.get('calibrated') is not None:
                report_lines.extend([
                    "🎯 CONFIDENCE CALIBRATION:",
                    f"Calibrated: {'✅ Yes' if calibration['calibrated'] else '❌ No'}",
                    f"Avg Error: {calibration.get('avg_calibration_error', 0):.1%}",
                    ""
                ])
            
            # Best conditions
            best_conditions = self.get_best_conditions()
            
            if best_conditions:
                report_lines.extend([
                    "⭐ TOP PERFORMING CONDITIONS:",
                    "-" * 30
                ])
                
                for i, condition in enumerate(best_conditions[:3], 1):
                    report_lines.append(
                        f"{i}. {condition['symbol']} ±{condition['target_pct']:.1%}: "
                        f"{condition['accuracy']:.1%} ({condition['predictions']} predictions)"
                    )
            
            return "\n".join(report_lines)
            
        except Exception as e:
            logger.error(f"Error generating analytics report: {e}")
            return "Error generating analytics report"


# Singleton instance
prediction_analytics = PredictionAnalytics()