# PHASE 5: TESTING & OPTIMIZATION (Days 12-14)
**🎯 GOAL**: Ensure system stability and optimize performance

## What You're Building
- Comprehensive end-to-end test suite
- Performance monitoring and optimization
- Production readiness validation
- Final system tuning and documentation

---

## CHECKPOINT 5A: End-to-End Testing (Day 12, 6 hours)

### Step 5.1: Comprehensive Test Suite
```python
# tests/test_end_to_end.py
import pytest
import asyncio
import time
from datetime import datetime, timedelta
import pandas as pd

class TestEndToEndSystem:
    """Comprehensive end-to-end system tests"""
    
    @pytest.fixture(scope="class")
    def system_components(self):
        """Setup system components for testing"""
        from src.utils.config import Config
        from src.database.operations import DatabaseOperations
        from src.data.fetcher import ExchangeDataFetcher
        from src.models.multi_target_trainer import MultiTargetModelTrainer
        from src.models.multi_target_predictor import MultiTargetPredictionEngine
        
        config = Config()
        db_ops = DatabaseOperations(config.database_url)
        fetcher = ExchangeDataFetcher()
        trainer = MultiTargetModelTrainer(db_ops)
        predictor = MultiTargetPredictionEngine(db_ops)
        predictor.trainer = trainer
        
        return {
            'config': config,
            'db': db_ops,
            'fetcher': fetcher,
            'trainer': trainer,
            'predictor': predictor
        }
    
    def test_data_pipeline(self, system_components):
        """Test complete data pipeline"""
        print("🧪 Testing data pipeline...")
        
        fetcher = system_components['fetcher']
        db = system_components['db']
        
        # Test data fetching
        btc_data = fetcher.fetch_historical_data('BTC/USDT', days=2)
        assert len(btc_data) > 24, f"Expected >24 candles, got {len(btc_data)}"
        
        # Test data saving
        if len(btc_data) > 0:
            candles = btc_data[['timestamp', 'open', 'high', 'low', 'close', 'volume']].values.tolist()
            candles = [[int(row[0].timestamp() * 1000)] + list(row[1:]) for row in candles]
            saved = db.save_ohlcv_data('BTC/USDT', '1h', candles)
            assert saved >= 0, "Data saving failed"
        
        # Test data retrieval
        retrieved = db.get_latest_candles('BTC/USDT', limit=50)
        assert len(retrieved) > 0, "Data retrieval failed"
        
        print("✅ Data pipeline test passed")
    
    def test_feature_engineering(self, system_components):
        """Test feature engineering pipeline"""
        print("🧪 Testing feature engineering...")
        
        from src.data.feature_engineer import MultiTargetFeatureEngineer
        
        db = system_components['db']
        feature_engineer = MultiTargetFeatureEngineer()
        
        # Get test data
        df = db.get_latest_candles('BTC/USDT', limit=200)
        assert len(df) >= 50, f"Need at least 50 candles for testing, got {len(df)}"
        
        # Test feature creation
        features = feature_engineer.create_features(df)
        assert features is not None, "Feature creation failed"
        assert len(features) > 20, f"Expected >20 features, got {len(features)}"
        
        # Test dataset preparation
        datasets = feature_engineer.prepare_training_datasets(df)
        assert len(datasets) > 0, "Dataset preparation failed"
        
        for target_pct, (X, y) in datasets.items():
            if X is not None:
                assert len(X) > 10, f"Dataset for {target_pct} too small: {len(X)}"
                assert len(X.columns) > 10, f"Too few features for {target_pct}: {len(X.columns)}"
        
        print("✅ Feature engineering test passed")
    
    def test_model_training(self, system_components):
        """Test model training pipeline"""
        print("🧪 Testing model training...")
        
        trainer = system_components['trainer']
        
        # Test training for one symbol
        results = trainer.train_models_for_symbol('BTC/USDT')
        
        # Check that at least one model trained successfully
        successful_models = sum(1 for success in results.values() if success)
        assert successful_models > 0, f"No models trained successfully: {results}"
        
        # Test model saving/loading
        for target_pct, success in results.items():
            if success:
                model_data = trainer.load_model_from_disk('BTC/USDT', target_pct)
                assert model_data is not None, f"Failed to load {target_pct} model"
                assert 'model' in model_data, "Model data missing 'model' key"
                assert 'feature_cols' in model_data, "Model data missing feature columns"
        
        print("✅ Model training test passed")
    
    def test_prediction_pipeline(self, system_components):
        """Test prediction pipeline"""
        print("🧪 Testing prediction pipeline...")
        
        trainer = system_components['trainer']
        predictor = system_components['predictor']
        
        # Load existing models
        for target_pct in [0.01, 0.02, 0.05]:
            model_data = trainer.load_model_from_disk('BTC/USDT', target_pct)
            if model_data:
                trainer.models[('BTC/USDT', target_pct)] = model_data
        
        # Test predictions
        predictions = predictor.predict_symbol_all_targets('BTC/USDT')
        
        assert 'symbol' in predictions, "Prediction missing symbol"
        assert 'predictions' in predictions, "Prediction missing predictions dict"
        
        # Check individual target predictions
        for target_str, pred_data in predictions['predictions'].items():
            if 'error' not in pred_data:
                assert 'prediction' in pred_data, f"Missing prediction for {target_str}"
                assert 'confidence' in pred_data, f"Missing confidence for {target_str}"
                assert pred_data['prediction'] in ['UP', 'DOWN'], f"Invalid prediction: {pred_data['prediction']}"
                assert 0 <= pred_data['confidence'] <= 1, f"Invalid confidence: {pred_data['confidence']}"
        
        print("✅ Prediction pipeline test passed")
    
    @pytest.mark.asyncio
    async def test_telegram_integration(self, system_components):
        """Test Telegram integration"""
        print("🧪 Testing Telegram integration...")
        
        # Skip if no Telegram credentials
        import os
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if not bot_token or not chat_id:
            print("⚠️ Skipping Telegram test - no credentials provided")
            return
        
        from src.notifications.telegram_reporter import MultiTargetTelegramReporter
        
        reporter = MultiTargetTelegramReporter(bot_token, chat_id)
        
        # Test connection
        connection_success = await reporter.test_connection()
        assert connection_success, "Telegram connection failed"
        
        # Test report sending with mock data
        mock_predictions = {
            'BTC/USDT': {
                'timestamp': datetime.now().isoformat(),
                'predictions': {
                    '1.0%': {
                        'prediction': 'UP',
                        'up_probability': 0.65,
                        'confidence': 0.65,
                        'model_accuracy': 0.62
                    }
                }
            }
        }
        
        report_success = await reporter.send_hourly_report(mock_predictions)
        assert report_success, "Failed to send test report"
        
        print("✅ Telegram integration test passed")
    
    def test_database_performance(self, system_components):
        """Test database performance with larger datasets"""
        print("🧪 Testing database performance...")
        
        db = system_components['db']
        
        # Test large data insertion
        start_time = time.time()
        
        # Generate mock OHLCV data
        mock_candles = []
        base_time = int(time.time() * 1000)
        
        for i in range(1000):  # 1000 candles
            timestamp = base_time + (i * 3600000)  # 1 hour intervals
            mock_candles.append([
                timestamp, 50000, 50100, 49900, 50050, 100.5  # OHLCV
            ])
        
        saved = db.save_ohlcv_data('TEST/USDT', '1h', mock_candles)
        insert_time = time.time() - start_time
        
        assert saved >= 0, "Bulk insert failed"
        assert insert_time < 10, f"Insert too slow: {insert_time:.2f}s for {len(mock_candles)} candles"
        
        # Test large data retrieval
        start_time = time.time()
        retrieved = db.get_latest_candles('TEST/USDT', limit=500)
        query_time = time.time() - start_time
        
        assert len(retrieved) > 0, "Data retrieval failed"
        assert query_time < 5, f"Query too slow: {query_time:.2f}s for {len(retrieved)} candles"
        
        print(f"✅ Database performance test passed (insert: {insert_time:.2f}s, query: {query_time:.2f}s)")
    
    def test_system_reliability(self, system_components):
        """Test system reliability under various conditions"""
        print("🧪 Testing system reliability...")
        
        predictor = system_components['predictor']
        
        # Test with insufficient data
        predictions = predictor.predict_symbol_all_targets('NONEXISTENT/USDT')
        assert 'error' in predictions, "Should fail gracefully with non-existent symbol"
        
        print("✅ System reliability test passed")

# Integration test runner
def run_integration_tests():
    """Run all integration tests"""
    print("🧪 RUNNING COMPREHENSIVE INTEGRATION TESTS")
    print("=" * 60)
    
    # Import pytest and run tests
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--color=yes"
    ])

if __name__ == "__main__":
    run_integration_tests()
```

### Step 5.2: Performance Monitoring Script
```python
# scripts/monitor_performance.py
import time
import psutil
import requests
import logging
from datetime import datetime, timedelta
import json

class PerformanceMonitor:
    def __init__(self, health_url="http://localhost:8000/health"):
        self.health_url = health_url
        self.metrics_history = []
        
    def collect_system_metrics(self):
        """Collect system performance metrics"""
        return {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'load_average': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0,
            'docker_stats': self.get_docker_stats()
        }
    
    def get_docker_stats(self):
        """Get Docker container statistics"""
        try:
            import subprocess
            result = subprocess.run([
                'docker', 'stats', '--no-stream', '--format', 
                'table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                containers = {}
                for line in lines:
                    parts = line.split('\t')
                    if len(parts) >= 3:
                        containers[parts[0]] = {
                            'cpu_percent': parts[1],
                            'memory_usage': parts[2]
                        }
                return containers
        except Exception as e:
            logging.error(f"Error getting Docker stats: {e}")
        
        return {}
    
    def check_application_health(self):
        """Check application health endpoint"""
        try:
            response = requests.get(self.health_url, timeout=10)
            return {
                'status_code': response.status_code,
                'response_time_ms': response.elapsed.total_seconds() * 1000,
                'health_data': response.json() if response.status_code == 200 else None
            }
        except Exception as e:
            return {
                'status_code': 0,
                'response_time_ms': 0,
                'error': str(e)
            }
    
    def run_monitoring_cycle(self):
        """Run one monitoring cycle"""
        print(f"🔍 Monitoring cycle at {datetime.now().strftime('%H:%M:%S')}")
        
        # Collect metrics
        system_metrics = self.collect_system_metrics()
        health_check = self.check_application_health()
        
        # Combine metrics
        full_metrics = {
            'system': system_metrics,
            'application': health_check
        }
        
        self.metrics_history.append(full_metrics)
        
        # Keep only last 24 hours of data
        cutoff_time = datetime.now() - timedelta(hours
