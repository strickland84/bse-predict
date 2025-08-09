"""Tests for prediction tracking and outcome monitoring."""
import unittest
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import json

from src.database.prediction_tracker import PredictionTracker
from src.tracking.outcome_monitor import OutcomeMonitor
from src.analytics.prediction_analytics import PredictionAnalytics
from src.models.multi_target_predictor import MultiTargetPredictionEngine


class TestPredictionTracking(unittest.TestCase):
    """Test prediction tracking functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tracker = PredictionTracker()
        # Mock the database connection
        self.tracker.conn = Mock()
        
    def test_save_prediction(self):
        """Test saving a single prediction."""
        # Mock successful insert
        self.tracker.conn.execute_query = Mock(return_value=[(123,)])
        
        # Test data
        symbol = "BTC/USDT"
        target_pct = 0.02
        timestamp = datetime.now(timezone.utc)
        model_name = "RandomForest_2.0%"
        prediction_class = 1  # UP
        probability = 0.75
        confidence = 0.75
        features_used = {"rsi": 65, "ema_ratio": 1.02}
        
        # Save prediction
        prediction_id = self.tracker.save_prediction(
            symbol, target_pct, timestamp, model_name,
            prediction_class, probability, confidence, features_used
        )
        
        # Verify
        self.assertEqual(prediction_id, 123)
        self.tracker.conn.execute_query.assert_called_once()
        
        # Check query parameters
        call_args = self.tracker.conn.execute_query.call_args
        self.assertEqual(call_args[0][1][0], symbol)
        self.assertEqual(call_args[0][1][1], target_pct)
        self.assertEqual(call_args[0][1][4], prediction_class)
        
    def test_save_predictions_batch(self):
        """Test saving multiple predictions."""
        # Mock successful inserts
        self.tracker.conn.execute_query = Mock(side_effect=[
            [(1,)], [(2,)], [(3,)]
        ])
        
        # Test data
        predictions = [
            {
                'symbol': 'BTC/USDT',
                'target_pct': 0.01,
                'timestamp': datetime.now(timezone.utc),
                'model_name': 'RandomForest_1.0%',
                'prediction_class': 1,
                'probability': 0.65,
                'confidence': 0.65,
                'features_used': {}
            },
            {
                'symbol': 'ETH/USDT',
                'target_pct': 0.02,
                'timestamp': datetime.now(timezone.utc),
                'model_name': 'RandomForest_2.0%',
                'prediction_class': 0,
                'probability': 0.70,
                'confidence': 0.70,
                'features_used': {}
            },
            {
                'symbol': 'SOL/USDT',
                'target_pct': 0.05,
                'timestamp': datetime.now(timezone.utc),
                'model_name': 'RandomForest_5.0%',
                'prediction_class': 1,
                'probability': 0.80,
                'confidence': 0.80,
                'features_used': {}
            }
        ]
        
        # Save batch
        prediction_ids = self.tracker.save_predictions_batch(predictions)
        
        # Verify
        self.assertEqual(len(prediction_ids), 3)
        self.assertEqual(prediction_ids, [1, 2, 3])
        self.assertEqual(self.tracker.conn.execute_query.call_count, 3)
        
    def test_get_pending_predictions(self):
        """Test retrieving pending predictions."""
        # Mock database response
        mock_df = pd.DataFrame({
            'id': [1, 2, 3],
            'symbol': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
            'target_pct': [0.01, 0.02, 0.05],
            'timestamp': [datetime.now(timezone.utc)] * 3,
            'prediction_class': [1, 0, 1],
            'probability': [0.65, 0.70, 0.80],
            'confidence': [0.65, 0.70, 0.80]
        })
        
        with patch('pandas.read_sql_query', return_value=mock_df):
            pending = self.tracker.get_pending_predictions(window_hours=24)
            
        # Verify
        self.assertEqual(len(pending), 3)
        self.assertIn('BTC/USDT', pending['symbol'].values)
        
    def test_update_prediction_outcome(self):
        """Test updating prediction outcome."""
        # Mock successful update
        self.tracker.conn.execute_query = Mock()
        
        # Update outcome
        success = self.tracker.update_prediction_outcome(
            prediction_id=1,
            actual_outcome=1,
            target_hit_timestamp=datetime.now(timezone.utc),
            time_to_target_hours=2.5,
            max_favorable_move=2.1,
            max_adverse_move=0.3
        )
        
        # Verify
        self.assertTrue(success)
        self.tracker.conn.execute_query.assert_called_once()


class TestOutcomeMonitor(unittest.TestCase):
    """Test outcome monitoring functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.monitor = OutcomeMonitor(monitoring_window_hours=72)
        # Mock database operations
        self.monitor.db_ops = Mock()
        
    @patch('src.tracking.outcome_monitor.prediction_tracker')
    def test_check_prediction_outcomes(self, mock_tracker):
        """Test checking prediction outcomes."""
        # Mock pending predictions
        pending_df = pd.DataFrame({
            'id': [1, 2],
            'symbol': ['BTC/USDT', 'ETH/USDT'],
            'target_pct': [0.02, 0.01],
            'timestamp': [datetime.now(timezone.utc) - timedelta(hours=2)] * 2,
            'prediction_class': [1, 0]  # UP, DOWN
        })
        
        mock_tracker.get_pending_predictions.return_value = pending_df
        
        # Mock price data for outcome checking
        price_data = pd.DataFrame({
            'timestamp': pd.date_range(
                start=datetime.now(timezone.utc) - timedelta(hours=3),
                periods=4,
                freq='H'
            ),
            'close': [50000, 50500, 51000, 51500],  # 3% increase
            'high': [50100, 50600, 51100, 51600],
            'low': [49900, 50400, 50900, 51400]
        })
        
        self.monitor.db_ops.get_candles_since.return_value = price_data
        mock_tracker.update_prediction_outcome.return_value = True
        
        # Check outcomes
        with patch.object(self.monitor, '_check_single_prediction') as mock_check:
            mock_check.side_effect = [
                {'completed': True, 'hit_target': True},
                {'completed': True, 'hit_target': False}
            ]
            
            stats = self.monitor.check_prediction_outcomes()
        
        # Verify
        self.assertEqual(stats['checked'], 2)
        self.assertEqual(stats['completed'], 2)
        self.assertEqual(stats['hit_target'], 1)
        self.assertEqual(stats['missed_target'], 1)
        
    def test_check_single_prediction_up_hit(self):
        """Test checking a single UP prediction that hits target."""
        # Create prediction
        prediction = pd.Series({
            'id': 1,
            'symbol': 'BTC/USDT',
            'target_pct': 0.02,  # 2% target
            'timestamp': datetime.now(timezone.utc) - timedelta(hours=2),
            'prediction_class': 1  # UP
        })
        
        # Mock price data - price goes up 2.5%
        price_data = pd.DataFrame({
            'timestamp': pd.date_range(
                start=prediction['timestamp'],
                periods=3,
                freq='H'
            ),
            'close': [50000, 50500, 51250],  # 2.5% increase
            'high': [50100, 50600, 51300],
            'low': [49900, 50400, 51200]
        })
        
        self.monitor.db_ops.get_candles_since.return_value = price_data
        
        # Check outcome
        with patch('src.tracking.outcome_monitor.prediction_tracker') as mock_tracker:
            mock_tracker.update_prediction_outcome.return_value = True
            outcome = self.monitor._check_single_prediction(prediction)
        
        # Verify
        self.assertTrue(outcome['completed'])
        self.assertTrue(outcome['hit_target'])
        self.assertIsNotNone(outcome['target_hit_timestamp'])
        self.assertGreater(outcome['max_favorable_move'], 2.0)
        
    def test_check_single_prediction_down_miss(self):
        """Test checking a single DOWN prediction that misses target."""
        # Create prediction
        prediction = pd.Series({
            'id': 2,
            'symbol': 'ETH/USDT',
            'target_pct': 0.05,  # 5% target
            'timestamp': datetime.now(timezone.utc) - timedelta(hours=80),  # Outside window
            'prediction_class': 0  # DOWN
        })
        
        # Mock price data - price goes up instead
        price_data = pd.DataFrame({
            'timestamp': pd.date_range(
                start=prediction['timestamp'],
                periods=73,  # Full monitoring window
                freq='H'
            ),
            'close': np.linspace(3000, 3100, 73),  # Price increases
            'high': np.linspace(3010, 3110, 73),
            'low': np.linspace(2990, 3090, 73)
        })
        
        self.monitor.db_ops.get_candles_since.return_value = price_data
        
        # Check outcome
        with patch('src.tracking.outcome_monitor.prediction_tracker') as mock_tracker:
            mock_tracker.update_prediction_outcome.return_value = True
            outcome = self.monitor._check_single_prediction(prediction)
        
        # Verify
        self.assertTrue(outcome['completed'])  # Window expired
        self.assertFalse(outcome['hit_target'])
        self.assertIsNone(outcome['target_hit_timestamp'])


class TestPredictionAnalytics(unittest.TestCase):
    """Test prediction analytics functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analytics = PredictionAnalytics()
        # Mock the tracker
        self.analytics.tracker = Mock()
        
    def test_get_overall_performance(self):
        """Test getting overall performance metrics."""
        # Mock performance data
        perf_df = pd.DataFrame({
            'symbol': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
            'target_pct': [0.01, 0.02, 0.05],
            'total_predictions': [100, 80, 60],
            'tracked_predictions': [90, 70, 50],
            'correct_predictions': [60, 50, 35],
            'accuracy': [0.667, 0.714, 0.700],
            'avg_confidence': [0.65, 0.70, 0.68],
            'avg_time_to_target': [3.5, 4.2, 5.1]
        })
        
        self.analytics.tracker.get_prediction_performance.return_value = perf_df
        
        # Get performance
        overall = self.analytics.get_overall_performance(days_back=7)
        
        # Verify
        self.assertEqual(overall['total_predictions'], 240)
        self.assertEqual(overall['tracked_predictions'], 210)
        self.assertAlmostEqual(overall['overall_accuracy'], 0.69, places=2)
        self.assertEqual(overall['best_performing']['symbol'], 'ETH/USDT')
        self.assertEqual(overall['worst_performing']['symbol'], 'BTC/USDT')
        
    def test_get_symbol_performance(self):
        """Test getting performance for specific symbol."""
        # Mock performance data
        perf_df = pd.DataFrame({
            'symbol': ['BTC/USDT', 'BTC/USDT', 'BTC/USDT'],
            'target_pct': [0.01, 0.02, 0.05],
            'total_predictions': [40, 30, 30],
            'tracked_predictions': [35, 25, 25],
            'correct_predictions': [25, 18, 15],
            'accuracy': [0.714, 0.720, 0.600],
            'avg_confidence': [0.65, 0.70, 0.75],
            'avg_time_to_target': [2.5, 4.0, 8.0]
        })
        
        self.analytics.tracker.get_prediction_performance.return_value = perf_df
        
        # Get performance
        symbol_perf = self.analytics.get_symbol_performance('BTC/USDT', days_back=7)
        
        # Verify
        self.assertEqual(symbol_perf['symbol'], 'BTC/USDT')
        self.assertEqual(symbol_perf['total_predictions'], 100)
        self.assertAlmostEqual(symbol_perf['overall_accuracy'], 0.682, places=2)
        self.assertIn('1.0%', symbol_perf['by_target'])
        self.assertEqual(symbol_perf['by_target']['1.0%']['accuracy'], 0.714)
        
    def test_confidence_calibration(self):
        """Test confidence calibration analysis."""
        # Mock recent predictions
        recent_preds = pd.DataFrame({
            'id': range(1, 21),
            'confidence': np.linspace(0.55, 0.95, 20),
            'status': ['Correct'] * 12 + ['Incorrect'] * 6 + ['Pending'] * 2
        })
        
        self.analytics.tracker.get_recent_predictions.return_value = recent_preds
        
        # Get calibration
        calibration = self.analytics.get_confidence_calibration()
        
        # Verify
        self.assertIn('calibrated', calibration)
        self.assertIn('avg_calibration_error', calibration)
        self.assertIn('buckets', calibration)


class TestPredictionEngine(unittest.TestCase):
    """Test prediction engine with tracking integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = MultiTargetPredictionEngine(
            target_percentages=[0.01, 0.02, 0.05],
            save_predictions=True
        )
        
    @patch('src.models.multi_target_predictor.prediction_tracker')
    @patch('src.models.multi_target_predictor.db_ops')
    def test_prediction_saving(self, mock_db_ops, mock_tracker):
        """Test that predictions are saved when enabled."""
        # Mock data and model
        mock_df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=200, freq='H'),
            'open': np.random.randn(200).cumsum() + 50000,
            'high': np.random.randn(200).cumsum() + 50100,
            'low': np.random.randn(200).cumsum() + 49900,
            'close': np.random.randn(200).cumsum() + 50000,
            'volume': np.random.rand(200) * 1000
        })
        
        mock_db_ops.get_latest_candles.return_value = mock_df
        
        # Mock feature engineering
        with patch.object(self.engine.feature_engineer, 'create_features') as mock_features:
            mock_features.return_value = {f'feature_{i}': np.random.rand() for i in range(10)}
            
            # Mock model loading
            with patch.object(self.engine.trainer, 'load_model_from_disk') as mock_load:
                mock_model = Mock()
                mock_model.predict_proba.return_value = [[0.3, 0.7]]  # 70% UP probability
                
                mock_load.return_value = {
                    'model': mock_model,
                    'feature_cols': [f'feature_{i}' for i in range(10)],
                    'accuracy': 0.65,
                    'cv_accuracy': 0.63
                }
                
                # Mock save prediction
                mock_tracker.save_prediction.return_value = 123
                
                # Make prediction
                result = self.engine.predict_symbol_all_targets('BTC/USDT')
                
        # Verify prediction was saved
        self.assertEqual(mock_tracker.save_prediction.call_count, 3)  # One for each target
        
        # Check first call arguments
        first_call = mock_tracker.save_prediction.call_args_list[0]
        self.assertEqual(first_call[1]['symbol'], 'BTC/USDT')
        self.assertEqual(first_call[1]['prediction_class'], 1)  # UP
        self.assertAlmostEqual(first_call[1]['probability'], 0.7, places=2)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete tracking system."""
    
    @patch('src.database.connection.DatabaseConnection')
    def test_full_prediction_cycle(self, mock_db_conn):
        """Test complete prediction and outcome tracking cycle."""
        # This would be a more comprehensive integration test
        # For now, just verify the components can be instantiated together
        
        tracker = PredictionTracker()
        monitor = OutcomeMonitor()
        analytics = PredictionAnalytics()
        
        self.assertIsNotNone(tracker)
        self.assertIsNotNone(monitor)
        self.assertIsNotNone(analytics)


if __name__ == '__main__':
    unittest.main()