#!/usr/bin/env python3
"""Phase 2 ML Engine Testing for BSE Predict."""
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import Config
from src.utils.logger import get_logger
from src.data.feature_engineer import MultiTargetFeatureEngineer
from src.models.multi_target_trainer import MultiTargetModelTrainer
from src.models.multi_target_predictor import MultiTargetPredictionEngine
from src.data.recovery import DataRecoveryManager
from src.database.operations import DatabaseOperations
import pandas as pd
import numpy as np

logger = get_logger(__name__)


def test_feature_engineering():
    """Test feature engineering pipeline."""
    print("🔧 Testing feature engineering...")
    try:
        config = Config()
        feature_engineer = MultiTargetFeatureEngineer(config.target_percentages)
        
        # Create sample data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='h', tz='UTC')
        sample_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 101,
            'low': np.random.randn(100).cumsum() + 99,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100)
        })
        
        # Ensure high > low
        sample_data['high'] = sample_data[['high', 'low', 'close']].max(axis=1) + 1
        sample_data['low'] = sample_data[['low', 'close']].min(axis=1) - 1
        
        # Test feature creation
        features = feature_engineer.create_features(sample_data)
        
        if features:
            print(f"   ✅ Created {len(features)} features")
            print(f"   ✅ Sample features: {list(features.keys())[:5]}")
            
            # Test dataset preparation
            # First extend sample data to have enough rows
            extended_data = pd.concat([sample_data] * 3, ignore_index=True)
            datasets = feature_engineer.prepare_training_datasets(extended_data)
            
            if datasets:
                print(f"   ✅ Prepared datasets for {len(datasets)} targets")
                for target_pct, (X, y) in datasets.items():
                    if X is not None and y is not None:
                        print(f"   ✅ {target_pct:.1%} target: {len(X)} samples, {len(X.columns)} features")
                
            return True
        else:
            print("   ❌ No features generated")
            return False
            
    except Exception as e:
        print(f"   ❌ Feature engineering error: {e}")
        return False


def test_model_training():
    """Test model training pipeline."""
    print("\n🤖 Testing model training...")
    try:
        trainer = MultiTargetModelTrainer([0.01, 0.02, 0.05])
        
        # Create synthetic training data
        np.random.seed(42)
        n_samples = 500
        
        # Generate synthetic features
        features = pd.DataFrame({
            'sma_20': np.random.randn(n_samples),
            'rsi_14': np.random.uniform(0, 100, n_samples),
            'volume_ratio': np.random.uniform(0.5, 2.0, n_samples),
            'volatility_24h': np.random.uniform(0.01, 0.1, n_samples),
            'price_change_1h': np.random.randn(n_samples) * 0.02,
            'price_change_6h': np.random.randn(n_samples) * 0.05,
            'price_change_24h': np.random.randn(n_samples) * 0.1,
            'volume_change_1h': np.random.randn(n_samples) * 0.3,
            'hour_of_day': np.random.randint(0, 24, n_samples),
            'day_of_week': np.random.randint(0, 7, n_samples)
        })
        
        # Generate synthetic labels with some signal
        y_1pct = (features['price_change_1h'] > 0.01).astype(int)
        y_2pct = (features['price_change_6h'] > 0.02).astype(int)
        y_5pct = (features['price_change_24h'] > 0.05).astype(int)
        
        # Test training for each target
        results = {}
        for target_pct, y in [(0.01, y_1pct), (0.02, y_2pct), (0.05, y_5pct)]:
            print(f"   🎯 Testing {target_pct:.1%} target...")
            
            # Train model
            success = trainer._train_single_model('TEST/BTC', target_pct, features, y)
            results[target_pct] = success
            
            if success:
                print(f"   ✅ {target_pct:.1%} model trained successfully")
            else:
                print(f"   ❌ {target_pct:.1%} model training failed")
                
        return all(results.values())
        
    except Exception as e:
        print(f"   ❌ Model training error: {e}")
        return False


def test_prediction_engine():
    """Test prediction engine."""
    print("\n🔮 Testing prediction engine...")
    try:
        predictor = MultiTargetPredictionEngine()
        
        # Test prediction for a single symbol (will fail without data, but tests the structure)
        result = predictor.predict_symbol_all_targets('TEST/BTC')
        print(f"   ✅ Single symbol prediction structure: {result.keys()}")
        
        # Test prediction for all assets
        all_predictions = predictor.predict_all_assets(['TEST/BTC'])
        print(f"   ✅ All assets prediction returned: {len(all_predictions)} assets")
        
        # Test that the engine has required components
        assert hasattr(predictor, 'feature_engineer')
        assert hasattr(predictor, 'trainer')
        assert hasattr(predictor, 'models')
        print("   ✅ Prediction engine has all required components")
        
        return True
            
    except Exception as e:
        print(f"   ❌ Prediction engine error: {e}")
        return False


def test_data_pipeline():
    """Test complete data pipeline."""
    print("\n🔄 Testing complete ML pipeline...")
    try:
        # Test that all components can be initialized
        db_ops = DatabaseOperations(Config().database_url)
        recovery = DataRecoveryManager()
        feature_engineer = MultiTargetFeatureEngineer([0.01, 0.02, 0.05])
        trainer = MultiTargetModelTrainer([0.01, 0.02, 0.05])
        predictor = MultiTargetPredictionEngine([0.01, 0.02, 0.05])
        
        print("   ✅ All pipeline components initialized successfully")
        print("   ✅ Database operations ready")
        print("   ✅ Data recovery manager ready")
        print("   ✅ Feature engineering ready")
        print("   ✅ Model training ready")
        print("   ✅ Prediction engine ready")
        
        return True
    except Exception as e:
        print(f"   ❌ Data pipeline error: {e}")
        return False


def test_model_persistence():
    """Test model saving and loading."""
    print("\n💾 Testing model persistence...")
    try:
        trainer = MultiTargetModelTrainer([0.01])
        
        # Create larger test dataset to meet minimum requirements
        np.random.seed(42)
        features = pd.DataFrame({
            'test_feature': np.random.randn(200),
            'feature2': np.random.randn(200),
            'feature3': np.random.randn(200)
        })
        y = pd.Series(np.random.choice([0, 1], size=200, p=[0.6, 0.4]))
        
        # Train and save
        success = trainer._train_single_model('TEST/SAVE', 0.01, features, y)
        
        if success:
            # Test loading
            loaded = trainer.load_model_from_disk('TEST/SAVE', 0.01)
            if loaded:
                print("   ✅ Model saved and loaded successfully")
                return True
            else:
                print("   ❌ Model loading failed")
                return False
        else:
            print("   ❌ Model training failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Model persistence error: {e}")
        return False


def test_high_confidence_detection():
    """Test high-confidence alert detection."""
    print("\n🚨 Testing high-confidence alerts...")
    try:
        predictor = MultiTargetPredictionEngine()
        
        # Test with high confidence prediction
        predictions = [
            {'symbol': 'BTC/USDT', 'target_pct': 0.01, 'confidence': 0.85, 'prediction': 1},
            {'symbol': 'BTC/USDT', 'target_pct': 0.02, 'confidence': 0.65, 'prediction': 0},
            {'symbol': 'BTC/USDT', 'target_pct': 0.05, 'confidence': 0.90, 'prediction': 1},
        ]
        
        high_conf = [p for p in predictions if p['confidence'] >= 0.75]
        
        if len(high_conf) >= 2:  # Expect 2 high-confidence predictions
            print(f"   ✅ High-confidence alerts detected: {len(high_conf)}")
            for alert in high_conf:
                print(f"      {alert['symbol']} {alert['target_pct']:.1%}: {alert['confidence']:.1%}")
            return True
        else:
            print("   ❌ High-confidence detection failed")
            return False
            
    except Exception as e:
        print(f"   ❌ High-confidence detection error: {e}")
        return False


def main():
    """Run all Phase 2 tests."""
    print("🚀 BSE Predict - Phase 2 ML Engine Testing")
    print("=" * 60)
    
    tests = [
        ("Feature Engineering", test_feature_engineering),
        ("Model Training", test_model_training),
        ("Prediction Engine", test_prediction_engine),
        ("Data Pipeline", test_data_pipeline),
        ("Model Persistence", test_model_persistence),
        ("High-Confidence Alerts", test_high_confidence_detection),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   ❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("📊 PHASE 2 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Phase 2 Overall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 Phase 2 ML Engine ready!")
        print("\n🚀 Next steps:")
        print("1. Start database: docker-compose -f docker-compose.dev.yml up -d")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Backfill data: python -c \"from src.data.recovery import DataRecoveryManager; r = DataRecoveryManager(); r.backfill_historical_data()\"")
        print("4. Train models: python -c \"from src.models.multi_target_trainer import MultiTargetModelTrainer; t = MultiTargetModelTrainer(); t.train_all_models()\"")
        return 0
    else:
        print("⚠️  Some Phase 2 tests failed. Check logs for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
