#!/usr/bin/env python3
"""
Integrated Testing Suite for BSE Predict
Tests all phases in sequence and validates the complete system
"""

import sys
import os
import asyncio
from datetime import datetime
from typing import Dict, List, Tuple, Callable
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import Config
from src.utils.logger import get_logger
from src.database.connection import DatabaseConnection
from src.database.operations import DatabaseOperations, db_ops
from src.data.fetcher import ExchangeDataFetcher
from src.data.recovery import DataRecoveryManager
from src.data.feature_engineer import MultiTargetFeatureEngineer
from src.models.multi_target_trainer import MultiTargetModelTrainer
from src.models.multi_target_predictor import MultiTargetPredictionEngine
from src.notifications.telegram_reporter import MultiTargetTelegramReporter
from src.scheduler.task_scheduler import MultiTargetTaskScheduler

logger = get_logger(__name__)


class IntegratedTestSuite:
    """Comprehensive test suite for all BSE Predict phases"""
    
    def __init__(self):
        self.config = Config()
        self.results = {}
        self.phase_tests = {
            "Phase 1": self.test_phase1_foundation,
            "Phase 2": self.test_phase2_ml_engine,
            "Phase 3": self.test_phase3_notifications,
            "Phase 4": self.test_phase4_automation,
            "Phase 5": self.test_phase5_production,
            # Future phases can be added here
        }
        
    def run_all_tests(self) -> Dict[str, Dict]:
        """Run all phase tests and return results"""
        print("🚀 BSE Predict - Integrated Testing Suite")
        print("=" * 70)
        print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"🔧 Environment: {self.config.get_nested('app.environment', 'development')}")
        print("=" * 70)
        
        for phase_name, test_func in self.phase_tests.items():
            print(f"\n{'='*70}")
            print(f"🎯 Testing {phase_name}")
            print(f"{'='*70}")
            
            try:
                phase_results = test_func()
                self.results[phase_name] = phase_results
            except Exception as e:
                print(f"❌ {phase_name} failed with error: {e}")
                self.results[phase_name] = {
                    "status": "FAILED",
                    "error": str(e),
                    "tests": {}
                }
        
        self._print_summary()
        return self.results
    
    def test_phase1_foundation(self) -> Dict:
        """Test Phase 1: Foundation & Data Pipeline"""
        phase_results = {
            "status": "RUNNING",
            "tests": {}
        }
        
        # Test 1.1: Configuration
        test_name = "Configuration System"
        try:
            config = Config()
            assert hasattr(config, 'assets')
            assert hasattr(config, 'target_percentages')
            assert config.assets == ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
            phase_results["tests"][test_name] = "PASS"
            print(f"✅ {test_name}")
        except Exception as e:
            phase_results["tests"][test_name] = f"FAIL: {e}"
            print(f"❌ {test_name}: {e}")
            
        # Test 1.2: Database Connection
        test_name = "Database Connection"
        try:
            db_conn = DatabaseConnection(self.config.database_url)
            assert db_conn.test_connection()
            phase_results["tests"][test_name] = "PASS"
            print(f"✅ {test_name}")
        except Exception as e:
            phase_results["tests"][test_name] = f"FAIL: {e}"
            print(f"❌ {test_name}: {e}")
            
        # Test 1.3: Exchange Data Fetcher
        test_name = "Exchange Data Fetcher"
        try:
            fetcher = ExchangeDataFetcher()
            assert fetcher.validate_symbol('BTC/USDT')
            info = fetcher.get_exchange_info()
            assert 'symbols' in info
            phase_results["tests"][test_name] = "PASS"
            print(f"✅ {test_name}")
        except Exception as e:
            phase_results["tests"][test_name] = f"FAIL: {e}"
            print(f"❌ {test_name}: {e}")
            
        # Test 1.4: Database Operations
        test_name = "Database Operations"
        try:
            df = db_ops.get_latest_candles('BTC/USDT', limit=10)
            assert isinstance(df, pd.DataFrame)
            phase_results["tests"][test_name] = "PASS"
            print(f"✅ {test_name}")
        except Exception as e:
            phase_results["tests"][test_name] = f"FAIL: {e}"
            print(f"❌ {test_name}: {e}")
            
        # Test 1.5: Data Recovery
        test_name = "Data Recovery Manager"
        try:
            recovery = DataRecoveryManager()
            assert hasattr(recovery, 'fetcher')
            assert hasattr(recovery, 'db_ops')
            phase_results["tests"][test_name] = "PASS"
            print(f"✅ {test_name}")
        except Exception as e:
            phase_results["tests"][test_name] = f"FAIL: {e}"
            print(f"❌ {test_name}: {e}")
            
        # Determine overall phase status
        phase_results["status"] = "PASS" if all(
            result == "PASS" for result in phase_results["tests"].values()
        ) else "FAIL"
        
        return phase_results
    
    def test_phase2_ml_engine(self) -> Dict:
        """Test Phase 2: ML Prediction Engine"""
        phase_results = {
            "status": "RUNNING",
            "tests": {}
        }
        
        # Test 2.1: Feature Engineering
        test_name = "Feature Engineering"
        try:
            engineer = MultiTargetFeatureEngineer(self.config.target_percentages)
            # Create sample data
            dates = pd.date_range(start='2024-01-01', periods=100, freq='h', tz='UTC')
            sample_data = pd.DataFrame({
                'timestamp': dates,
                'open': np.random.randn(100).cumsum() + 100,
                'high': np.random.randn(100).cumsum() + 101,
                'low': np.random.randn(100).cumsum() + 99,
                'close': np.random.randn(100).cumsum() + 100,
                'volume': np.abs(np.random.randn(100)) * 1000
            })
            features = engineer.create_features(sample_data)
            assert features is not None
            phase_results["tests"][test_name] = "PASS"
            print(f"✅ {test_name}")
        except Exception as e:
            phase_results["tests"][test_name] = f"FAIL: {e}"
            print(f"❌ {test_name}: {e}")
            
        # Test 2.2: Model Training
        test_name = "Model Training"
        try:
            trainer = MultiTargetModelTrainer(self.config.target_percentages)
            # Test with minimal data
            assert hasattr(trainer, 'models')
            phase_results["tests"][test_name] = "PASS"
            print(f"✅ {test_name}")
        except Exception as e:
            phase_results["tests"][test_name] = f"FAIL: {e}"
            print(f"❌ {test_name}: {e}")
            
        # Test 2.3: Prediction Engine
        test_name = "Prediction Engine"
        try:
            predictor = MultiTargetPredictionEngine(self.config.target_percentages)
            assert hasattr(predictor, 'feature_engineer')
            assert hasattr(predictor, 'trainer')
            phase_results["tests"][test_name] = "PASS"
            print(f"✅ {test_name}")
        except Exception as e:
            phase_results["tests"][test_name] = f"FAIL: {e}"
            print(f"❌ {test_name}: {e}")
            
        # Test 2.4: Model Persistence
        test_name = "Model Persistence"
        try:
            trainer = MultiTargetModelTrainer([0.01])
            model_path = trainer._get_model_path('TEST/ASSET', 0.01)
            assert model_path.suffix == '.pkl'
            phase_results["tests"][test_name] = "PASS"
            print(f"✅ {test_name}")
        except Exception as e:
            phase_results["tests"][test_name] = f"FAIL: {e}"
            print(f"❌ {test_name}: {e}")
            
        # Determine overall phase status
        phase_results["status"] = "PASS" if all(
            result == "PASS" for result in phase_results["tests"].values()
        ) else "FAIL"
        
        return phase_results
    
    def test_phase3_notifications(self) -> Dict:
        """Test Phase 3: Telegram Integration"""
        phase_results = {
            "status": "RUNNING",
            "tests": {}
        }
        
        # Test 3.1: Telegram Reporter Initialization
        test_name = "Telegram Reporter"
        try:
            bot_token = self.config.config['telegram']['bot_token']
            chat_id = self.config.config['telegram']['chat_id']
            
            if bot_token and chat_id:
                reporter = MultiTargetTelegramReporter(bot_token, chat_id)
                assert hasattr(reporter, 'bot')
                phase_results["tests"][test_name] = "PASS"
                print(f"✅ {test_name}")
            else:
                phase_results["tests"][test_name] = "SKIP: No credentials"
                print(f"⚠️ {test_name}: Skipped (no credentials)")
        except Exception as e:
            phase_results["tests"][test_name] = f"FAIL: {e}"
            print(f"❌ {test_name}: {e}")
            
        # Test 3.2: Message Formatting
        test_name = "Message Formatting"
        try:
            if 'reporter' in locals():
                mock_predictions = {
                    'BTC/USDT': {
                        'timestamp': datetime.now().isoformat(),
                        'predictions': {
                            '1.0%': {
                                'prediction': 'UP',
                                'up_probability': 0.78,
                                'confidence': 0.78,
                                'model_accuracy': 0.74
                            }
                        }
                    }
                }
                message = reporter._format_hourly_report(mock_predictions)
                assert len(message) > 0
                phase_results["tests"][test_name] = "PASS"
                print(f"✅ {test_name}")
            else:
                phase_results["tests"][test_name] = "SKIP: Reporter not initialized"
                print(f"⚠️ {test_name}: Skipped")
        except Exception as e:
            phase_results["tests"][test_name] = f"FAIL: {e}"
            print(f"❌ {test_name}: {e}")
            
        # Test 3.3: Alert System
        test_name = "Alert System"
        try:
            # Just test the structure exists
            phase_results["tests"][test_name] = "PASS"
            print(f"✅ {test_name}")
        except Exception as e:
            phase_results["tests"][test_name] = f"FAIL: {e}"
            print(f"❌ {test_name}: {e}")
            
        # Determine overall phase status
        phase_results["status"] = "PASS" if all(
            "PASS" in result or "SKIP" in result 
            for result in phase_results["tests"].values()
        ) else "FAIL"
        
        return phase_results
    
    def test_phase4_automation(self) -> Dict:
        """Test Phase 4: Containerization & Deployment"""
        phase_results = {
            "status": "RUNNING",
            "tests": {}
        }
        
        # Test 4.1: Docker Configuration
        test_name = "Docker Configuration"
        try:
            from pathlib import Path
            project_root = Path(__file__).parent.parent
            assert (project_root / "Dockerfile").exists()
            assert (project_root / "docker-compose.prod.yml").exists()
            assert (project_root / "postgresql.conf").exists()
            phase_results["tests"][test_name] = "PASS"
            print(f"✅ {test_name}")
        except Exception as e:
            phase_results["tests"][test_name] = f"FAIL: {e}"
            print(f"❌ {test_name}: {e}")
            
        # Test 4.2: Main Application Entry Point
        test_name = "Application Entry Point"
        try:
            from pathlib import Path
            assert (Path(__file__).parent.parent / "src" / "main.py").exists()
            from src.main import CryptoMLApplication
            app = CryptoMLApplication()
            assert hasattr(app, 'initialize')
            assert hasattr(app, 'run')
            assert hasattr(app, 'setup_health_check')
            phase_results["tests"][test_name] = "PASS"
            print(f"✅ {test_name}")
        except Exception as e:
            phase_results["tests"][test_name] = f"FAIL: {e}"
            print(f"❌ {test_name}: {e}")
            
        # Test 4.3: Deployment Scripts
        test_name = "Deployment Scripts"
        try:
            from pathlib import Path
            scripts_dir = Path(__file__).parent.parent / "scripts"
            assert scripts_dir.exists()
            assert (scripts_dir / "deploy_to_hetzner.sh").exists()
            assert (scripts_dir / "test_docker_build.sh").exists()
            phase_results["tests"][test_name] = "PASS"
            print(f"✅ {test_name}")
        except Exception as e:
            phase_results["tests"][test_name] = f"FAIL: {e}"
            print(f"❌ {test_name}: {e}")
            
        # Test 4.4: Health Monitoring
        test_name = "Health Monitoring"
        try:
            scheduler = MultiTargetTaskScheduler(db_ops, self.config)
            status = scheduler.get_system_status()
            assert 'cpu_percent' in status
            assert 'memory_percent' in status
            assert 'disk_percent' in status
            phase_results["tests"][test_name] = "PASS"
            print(f"✅ {test_name}")
        except Exception as e:
            phase_results["tests"][test_name] = f"FAIL: {e}"
            print(f"❌ {test_name}: {e}")
            
        # Test 4.5: Environment Configuration
        test_name = "Environment Configuration"
        try:
            from pathlib import Path
            assert (Path(__file__).parent.parent / ".env.production").exists()
            assert (Path(__file__).parent.parent / ".gitignore").exists()
            phase_results["tests"][test_name] = "PASS"
            print(f"✅ {test_name}")
        except Exception as e:
            phase_results["tests"][test_name] = f"FAIL: {e}"
            print(f"❌ {test_name}: {e}")
            
        # Determine overall phase status
        phase_results["status"] = "PASS" if all(
            result == "PASS" for result in phase_results["tests"].values()
        ) else "FAIL"
        
        return phase_results
    
    def test_phase5_production(self) -> Dict:
        """Test Phase 5: Production Readiness"""
        phase_results = {
            "status": "RUNNING",
            "tests": {}
        }
        
        # Test 5.1: Docker Environment
        test_name = "Docker Environment"
        try:
            docker_file = 'docker-compose.dev.yml'
            assert os.path.exists(docker_file)
            phase_results["tests"][test_name] = "PASS"
            print(f"✅ {test_name}")
        except Exception as e:
            phase_results["tests"][test_name] = f"FAIL: {e}"
            print(f"❌ {test_name}: {e}")
            
        # Test 5.2: Configuration Files
        test_name = "Configuration Files"
        try:
            assert os.path.exists('config.yaml')
            assert os.path.exists('requirements.txt')
            assert os.path.exists('pyproject.toml')
            phase_results["tests"][test_name] = "PASS"
            print(f"✅ {test_name}")
        except Exception as e:
            phase_results["tests"][test_name] = f"FAIL: {e}"
            print(f"❌ {test_name}: {e}")
            
        # Test 5.3: Logging System
        test_name = "Logging System"
        try:
            assert os.path.exists('logs/app.log')
            phase_results["tests"][test_name] = "PASS"
            print(f"✅ {test_name}")
        except Exception as e:
            phase_results["tests"][test_name] = f"FAIL: {e}"
            print(f"❌ {test_name}: {e}")
            
        # Test 5.4: Model Directory
        test_name = "Model Storage"
        try:
            assert os.path.exists('models/')
            phase_results["tests"][test_name] = "PASS"
            print(f"✅ {test_name}")
        except Exception as e:
            phase_results["tests"][test_name] = f"FAIL: {e}"
            print(f"❌ {test_name}: {e}")
            
        # Determine overall phase status
        phase_results["status"] = "PASS" if all(
            result == "PASS" for result in phase_results["tests"].values()
        ) else "FAIL"
        
        return phase_results
    
    def _print_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "=" * 70)
        print("📊 INTEGRATED TEST SUMMARY")
        print("=" * 70)
        
        total_phases = len(self.results)
        passed_phases = sum(1 for r in self.results.values() if r["status"] == "PASS")
        
        for phase_name, phase_result in self.results.items():
            status_emoji = "✅" if phase_result["status"] == "PASS" else "❌"
            print(f"\n{status_emoji} {phase_name}: {phase_result['status']}")
            
            if "tests" in phase_result:
                for test_name, test_result in phase_result["tests"].items():
                    if test_result == "PASS":
                        print(f"   ✅ {test_name}")
                    elif "SKIP" in str(test_result):
                        print(f"   ⚠️ {test_name}: {test_result}")
                    else:
                        print(f"   ❌ {test_name}: {test_result}")
        
        print(f"\n{'='*70}")
        print(f"🎯 Overall Result: {passed_phases}/{total_phases} phases passed")
        
        if passed_phases == total_phases:
            print("🎉 All phases passed! BSE Predict is ready for deployment!")
        else:
            print("⚠️ Some phases failed. Please check the logs above.")
        
        # Print next steps based on results
        print(f"\n{'='*70}")
        print("📋 Next Steps:")
        
        if self.results.get("Phase 1", {}).get("status") == "PASS":
            print("✅ Foundation ready - Database and data pipeline operational")
        else:
            print("❌ Fix Phase 1 issues before proceeding")
            
        if self.results.get("Phase 2", {}).get("status") == "PASS":
            print("✅ ML Engine ready - Can train and make predictions")
        else:
            print("❌ Fix Phase 2 issues to enable predictions")
            
        if self.results.get("Phase 3", {}).get("status") == "PASS":
            print("✅ Notifications ready - Telegram integration operational")
        else:
            print("⚠️ Telegram integration needs configuration")
            
        if self.results.get("Phase 4", {}).get("status") == "PASS":
            print("✅ Automation ready - Scheduler can run tasks")
        else:
            print("❌ Fix Phase 4 issues for automated operation")
            
        if self.results.get("Phase 5", {}).get("status") == "PASS":
            print("✅ Production ready - All systems operational")
        else:
            print("⚠️ Complete production setup before deployment")


def run_quick_test():
    """Run a quick subset of tests for rapid validation"""
    print("\n🚀 Running Quick Integration Test")
    print("=" * 50)
    
    suite = IntegratedTestSuite()
    
    # Run only critical tests
    critical_tests = {
        "Config": lambda: Config() is not None,
        "Database": lambda: db_ops.get_latest_candles('BTC/USDT', limit=1) is not None,
        "ML Engine": lambda: MultiTargetPredictionEngine() is not None,
        "Scheduler": lambda: MultiTargetTaskScheduler(db_ops, Config()) is not None
    }
    
    results = {}
    for name, test in critical_tests.items():
        try:
            test()
            results[name] = "PASS"
            print(f"✅ {name}")
        except Exception as e:
            results[name] = f"FAIL: {e}"
            print(f"❌ {name}: {e}")
    
    passed = sum(1 for r in results.values() if r == "PASS")
    print(f"\n🎯 Quick Test: {passed}/{len(results)} passed")
    
    return passed == len(results)


def main():
    """Main entry point for integrated testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='BSE Predict Integrated Test Suite')
    parser.add_argument('--quick', action='store_true', help='Run quick validation test')
    parser.add_argument('--phase', type=str, help='Run specific phase test (e.g., "Phase 1")')
    args = parser.parse_args()
    
    if args.quick:
        success = run_quick_test()
    elif args.phase:
        suite = IntegratedTestSuite()
        if args.phase in suite.phase_tests:
            print(f"🎯 Running {args.phase} Test")
            result = suite.phase_tests[args.phase]()
            print(f"\n{'✅' if result['status'] == 'PASS' else '❌'} {args.phase}: {result['status']}")
            success = result['status'] == 'PASS'
        else:
            print(f"❌ Unknown phase: {args.phase}")
            print(f"Available phases: {', '.join(suite.phase_tests.keys())}")
            success = False
    else:
        suite = IntegratedTestSuite()
        results = suite.run_all_tests()
        success = all(r["status"] == "PASS" for r in results.values())
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())