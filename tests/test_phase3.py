#!/usr/bin/env python3
"""
Phase 3 Testing Script - Telegram Integration & Automation
Tests all components of the notification and scheduling system
"""

import asyncio
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.notifications.telegram_reporter import MultiTargetTelegramReporter
from src.scheduler.task_scheduler import MultiTargetTaskScheduler
from src.database.operations import DatabaseOperations
from src.utils.config import Config

def test_telegram_connection():
    """Test Telegram bot connection"""
    print("🧪 Testing Telegram Connection...")
    
    try:
        config = Config()
        bot_token = config.config['telegram']['bot_token']
        chat_id = config.config['telegram']['chat_id']
        
        if not bot_token or not chat_id:
            print("❌ Telegram credentials not found in config")
            return False
            
        reporter = MultiTargetTelegramReporter(bot_token, chat_id)
        
        # Test connection
        success = asyncio.run(reporter.test_connection())
        
        if success:
            print("✅ Telegram connection successful")
            return True
        else:
            print("❌ Telegram connection failed")
            return False
            
    except Exception as e:
        print(f"❌ Telegram test error: {e}")
        return False

def test_telegram_formatting():
    """Test message formatting"""
    print("\n🧪 Testing Telegram Message Formatting...")
    
    try:
        config = Config()
        bot_token = config.config['telegram']['bot_token']
        chat_id = config.config['telegram']['chat_id']
        
        if not bot_token or not chat_id:
            print("⚠️ Skipping formatting test - no Telegram credentials")
            return True
            
        reporter = MultiTargetTelegramReporter(bot_token, chat_id)
        
        # Mock prediction data
        mock_predictions = {
            'BTC/USDT': {
                'timestamp': datetime.now().isoformat(),
                'predictions': {
                    '1.0%': {
                        'prediction': 'UP',
                        'up_probability': 0.78,
                        'confidence': 0.78,
                        'model_accuracy': 0.74
                    },
                    '2.0%': {
                        'prediction': 'DOWN',
                        'up_probability': 0.32,
                        'confidence': 0.68,
                        'model_accuracy': 0.72
                    },
                    '5.0%': {
                        'prediction': 'UP',
                        'up_probability': 0.85,
                        'confidence': 0.85,
                        'model_accuracy': 0.69
                    }
                }
            },
            'ETH/USDT': {
                'timestamp': datetime.now().isoformat(),
                'predictions': {
                    '1.0%': {
                        'prediction': 'DOWN',
                        'up_probability': 0.25,
                        'confidence': 0.75,
                        'model_accuracy': 0.71
                    },
                    '2.0%': {
                        'error': 'No trained model for 2.0% target'
                    },
                    '5.0%': {
                        'prediction': 'UP',
                        'up_probability': 0.65,
                        'confidence': 0.65,
                        'model_accuracy': 0.68
                    }
                }
            }
        }
        
        # Run both tests in the same event loop
        async def run_tests():
            # Test hourly report
            success = await reporter.send_hourly_report(mock_predictions)
            
            if success:
                print("✅ Hourly report formatting successful")
            else:
                print("❌ Hourly report formatting failed")
                
            # Test high-confidence alert
            alert_success = await reporter.send_high_confidence_alert(
                'BTC/USDT', '5.0%', {
                    'prediction': 'UP',
                    'confidence': 0.85,
                    'up_probability': 0.85
                }
            )
            
            if alert_success:
                print("✅ High-confidence alert successful")
            else:
                print("❌ High-confidence alert failed")
                
            return success and alert_success
        
        return asyncio.run(run_tests())
        
    except Exception as e:
        print(f"❌ Formatting test error: {e}")
        return False

def test_scheduler_initialization():
    """Test scheduler initialization"""
    print("\n🧪 Testing Scheduler Initialization...")
    
    try:
        config = Config()
        db_ops = DatabaseOperations(config.database_url)
        scheduler = MultiTargetTaskScheduler(db_ops, config)
        
        # Test initialization
        scheduler.initialize_components()
        
        # Test schedule setup
        scheduler.setup_schedules()
        
        print("✅ Scheduler initialization successful")
        return True
        
    except Exception as e:
        print(f"❌ Scheduler initialization error: {e}")
        return False

def test_manual_prediction():
    """Test manual prediction workflow"""
    print("\n🧪 Testing Manual Prediction...")
    
    try:
        config = Config()
        db_ops = DatabaseOperations(config.database_url)
        scheduler = MultiTargetTaskScheduler(db_ops, config)
        
        # Initialize components first
        scheduler.initialize_components()
        
        # Test manual prediction
        predictions = scheduler.run_manual_prediction()
        
        if predictions:
            print("✅ Manual prediction successful")
            print(f"   Generated predictions for {len(predictions)} assets")
            return True
        else:
            print("❌ Manual prediction failed")
            return False
            
    except Exception as e:
        print(f"❌ Manual prediction error: {e}")
        return False

def test_system_status():
    """Test system status reporting"""
    print("\n🧪 Testing System Status...")
    
    try:
        config = Config()
        db_ops = DatabaseOperations(config.database_url)
        scheduler = MultiTargetTaskScheduler(db_ops, config)
        
        # Test system status
        status = scheduler.get_system_status()
        
        print("✅ System status retrieved:")
        print(f"   CPU: {status['cpu_percent']:.1f}%")
        print(f"   Memory: {status['memory_percent']:.1f}%")
        print(f"   Disk: {status['disk_percent']:.1f}%")
        print(f"   Models: {status['active_models']}")
        
        return True
        
    except Exception as e:
        print(f"❌ System status error: {e}")
        return False

def main():
    """Run all Phase 3 tests"""
    print("🚀 Phase 3 Testing - Telegram Integration & Automation")
    print("=" * 60)
    
    tests = [
        ("Telegram Connection", test_telegram_connection),
        ("Message Formatting", test_telegram_formatting),
        ("Scheduler Initialization", test_scheduler_initialization),
        ("Manual Prediction", test_manual_prediction),
        ("System Status", test_system_status)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        result = test_func()
        results.append((test_name, result))
        
    # Summary
    print("\n" + "=" * 60)
    print("📊 Phase 3 Test Results:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All Phase 3 tests passed! Ready for deployment.")
    else:
        print("⚠️ Some tests failed. Check logs above.")
        
    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
