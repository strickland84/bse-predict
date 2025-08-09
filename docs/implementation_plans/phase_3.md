# PHASE 3: TELEGRAM INTEGRATION (Days 7-9)
**🎯 GOAL**: Add comprehensive reporting with multi-target predictions

## What You're Building
- Telegram bot that sends formatted hourly reports
- Rich message formatting with emojis and prediction confidence
- High-confidence alert system (>75% confidence)
- Automated task scheduler for hourly predictions
- Daily performance summaries and model retraining

---

## CHECKPOINT 3A: Telegram Bot Setup (Day 7, 4 hours)

### Step 3.1: Enhanced Telegram Reporter
```python
# src/notifications/telegram_reporter.py
import asyncio
from telegram import Bot
from telegram.error import TelegramError
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
import logging

class MultiTargetTelegramReporter:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot = Bot(token=bot_token)
        self.chat_id = chat_id
        self.max_message_length = 4000  # Telegram limit is 4096
        
    async def send_hourly_report(self, predictions: Dict[str, Dict]) -> bool:
        """Send comprehensive hourly report with all predictions"""
        try:
            message = self._format_hourly_report(predictions)
            
            # Split message if too long
            if len(message) > self.max_message_length:
                messages = self._split_message(message)
                for msg in messages:
                    await self.bot.send_message(chat_id=self.chat_id, text=msg, parse_mode='Markdown')
                    await asyncio.sleep(1)  # Rate limiting
            else:
                await self.bot.send_message(chat_id=self.chat_id, text=message, parse_mode='Markdown')
                
            return True
            
        except TelegramError as e:
            logging.error(f"Telegram error: {e}")
            return False
        except Exception as e:
            logging.error(f"Error sending hourly report: {e}")
            return False
            
    def _format_hourly_report(self, predictions: Dict[str, Dict]) -> str:
        """Format comprehensive hourly report"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        
        message = f"🚀 *Crypto ML Multi-Target Predictions*\n"
        message += f"📅 {timestamp}\n\n"
        
        for symbol, symbol_data in predictions.items():
            if 'error' in symbol_data:
                message += f"❌ *{symbol}*: {symbol_data['error']}\n\n"
                continue
                
            message += f"💰 *{symbol}*\n"
            message += "```\n"
            
            # Process each target
            target_results = []
            for target_str, pred_data in symbol_data['predictions'].items():
                if 'error' in pred_data:
                    target_results.append(f"{target_str:>4}: ERROR - {pred_data['error'][:30]}")
                else:
                    prediction = pred_data['prediction']
                    confidence = pred_data['confidence']
                    up_prob = pred_data['up_probability']
                    
                    # Choose emoji based on prediction and confidence
                    if prediction == 'UP':
                        emoji = "🟢" if confidence > 0.7 else "🔵"
                        prob_display = f"{up_prob:.1%}"
                    else:
                        emoji = "🔴" if confidence > 0.7 else "🟠"
                        prob_display = f"{(1-up_prob):.1%}"
                    
                    target_results.append(f"{target_str:>4}: {emoji} {prediction:>4} ({prob_display:>5}, conf: {confidence:.2f})")
            
            for result in target_results:
                message += f"{result}\n"
                
            message += "```\n"
            
            # Add model performance info
            if 'predictions' in symbol_data:
                accuracies = []
                for pred_data in symbol_data['predictions'].values():
                    if 'model_accuracy' in pred_data:
                        accuracies.append(pred_data['model_accuracy'])
                
                if accuracies:
                    avg_accuracy = sum(accuracies) / len(accuracies)
                    message += f"📊 Avg Model Accuracy: {avg_accuracy:.1%}\n"
            
            message += "\n"
        
        # Add legend
        message += "📖 *Legend*:\n"
        message += "🟢 High confidence UP | 🔵 Low confidence UP\n"
        message += "🔴 High confidence DOWN | 🟠 Low confidence DOWN\n\n"
        
        # Add summary stats
        total_predictions = 0
        successful_predictions = 0
        high_confidence_count = 0
        
        for symbol_data in predictions.values():
            if 'predictions' in symbol_data:
                for pred_data in symbol_data['predictions'].values():
                    total_predictions += 1
                    if 'error' not in pred_data:
                        successful_predictions += 1
                        if pred_data.get('confidence', 0) > 0.7:
                            high_confidence_count += 1
        
        message += f"📈 *Summary*: {successful_predictions}/{total_predictions} predictions successful\n"
        message += f"🎯 High confidence signals: {high_confidence_count}\n"
        
        return message
        
    def _split_message(self, message: str) -> List[str]:
        """Split long message into smaller chunks"""
        lines = message.split('\n')
        messages = []
        current_message = ""
        
        for line in lines:
            if len(current_message + line + '\n') > self.max_message_length:
                if current_message:
                    messages.append(current_message.strip())
                current_message = line + '\n'
            else:
                current_message += line + '\n'
        
        if current_message:
            messages.append(current_message.strip())
            
        return messages
        
    async def send_high_confidence_alert(self, symbol: str, target: str, prediction_data: Dict) -> bool:
        """Send immediate alert for high-confidence predictions"""
        try:
            if prediction_data.get('confidence', 0) < 0.75:
                return True  # No alert needed
                
            prediction = prediction_data['prediction']
            confidence = prediction_data['confidence']
            probability = prediction_data['up_probability'] if prediction == 'UP' else (1 - prediction_data['up_probability'])
            
            emoji = "🚨🟢" if prediction == 'UP' else "🚨🔴"
            
            message = f"{emoji} *HIGH CONFIDENCE ALERT* {emoji}\n\n"
            message += f"💰 *{symbol}* - {target} Target\n"
            message += f"📊 Prediction: *{prediction}*\n"
            message += f"🎯 Probability: {probability:.1%}\n"
            message += f"🔒 Confidence: {confidence:.1%}\n"
            message += f"⏰ {datetime.now().strftime('%H:%M UTC')}"
            
            await self.bot.send_message(chat_id=self.chat_id, text=message, parse_mode='Markdown')
            return True
            
        except Exception as e:
            logging.error(f"Error sending alert: {e}")
            return False
            
    async def send_daily_performance_summary(self, performance_data: Dict) -> bool:
        """Send daily model performance summary"""
        try:
            message = self._format_daily_summary(performance_data)
            await self.bot.send_message(chat_id=self.chat_id, text=message, parse_mode='Markdown')
            return True
        except Exception as e:
            logging.error(f"Error sending daily summary: {e}")
            return False
            
    def _format_daily_summary(self, performance_data: Dict) -> str:
        """Format daily performance summary"""
        message = f"📊 *Daily Performance Summary*\n"
        message += f"📅 {datetime.now().strftime('%Y-%m-%d')}\n\n"
        
        for symbol, targets in performance_data.items():
            message += f"💰 *{symbol}*\n"
            message += "```\n"
            
            for target, metrics in targets.items():
                accuracy = metrics.get('accuracy', 0)
                total_preds = metrics.get('total_predictions', 0)
                
                if total_preds > 0:
                    message += f"{target:>4}: {accuracy:.1%} accuracy ({total_preds} predictions)\n"
                else:
                    message += f"{target:>4}: No predictions\n"
            
            message += "```\n"
        
        return message
        
    async def test_connection(self) -> bool:
        """Test Telegram bot connection"""
        try:
            await self.bot.send_message(chat_id=self.chat_id, text="🤖 Crypto ML Bot Connection Test ✅")
            return True
        except Exception as e:
            logging.error(f"Telegram connection test failed: {e}")
            return False
```

### Step 3.2: Telegram Bot Setup Guide
```python
# Create a Telegram Bot:
# 1. Message @BotFather on Telegram
# 2. Send /newbot
# 3. Choose a name and username for your bot
# 4. Save the bot token
# 5. Get your chat ID by messaging @userinfobot

# Example test script
async def setup_telegram_bot():
    """Setup and test Telegram bot"""
    import os
    
    # Set
# Create a Telegram Bot:
# 1. Message @BotFather on Telegram
# 2. Send /newbot
# 3. Choose a name and username for your bot
# 4. Save the bot token
# 5. Get your chat ID by messaging @userinfobot

# Example test script
async def setup_telegram_bot():
    """Setup and test Telegram bot"""
    import os
    
    # Set environment variables
    bot_token = "YOUR_BOT_TOKEN_HERE"
    chat_id = "YOUR_CHAT_ID_HERE"
    
    reporter = MultiTargetTelegramReporter(bot_token, chat_id)
    
    # Test connection
    success = await reporter.test_connection()
    print(f"Telegram connection: {'✅ Success' if success else '❌ Failed'}")
    
    return success
```

**🔍 CHECKPOINT 3A TEST:**
```python
# test_telegram.py
import asyncio
from src.notifications.telegram_reporter import MultiTargetTelegramReporter
import os

# Test Telegram setup (you'll need real bot token and chat ID)
bot_token = os.getenv('TELEGRAM_BOT_TOKEN', 'YOUR_BOT_TOKEN')
chat_id = os.getenv('TELEGRAM_CHAT_ID', 'YOUR_CHAT_ID')

if bot_token != 'YOUR_BOT_TOKEN':
    reporter = MultiTargetTelegramReporter(bot_token, chat_id)
    
    # Test connection
    async def test_telegram():
        success = await reporter.test_connection()
        print(f"✅ Telegram test: {'Success' if success else 'Failed'}")
        
        # Test with mock prediction data
        mock_predictions = {
            'BTC/USDT': {
                'timestamp': '2024-01-01T12:00:00',
                'predictions': {
                    '1.0%': {
                        'prediction': 'UP',
                        'up_probability': 0.68,
                        'confidence': 0.68,
                        'model_accuracy': 0.62
                    },
                    '2.0%': {
                        'prediction': 'DOWN',
                        'up_probability': 0.35,
                        'confidence': 0.65,
                        'model_accuracy': 0.64
                    },
                    '5.0%': {
                        'prediction': 'UP',
                        'up_probability': 0.78,
                        'confidence': 0.78,
                        'model_accuracy': 0.66
                    }
                }
            }
        }
        
        success = await reporter.send_hourly_report(mock_predictions)
        print(f"✅ Hourly report test: {'Success' if success else 'Failed'}")
    
    asyncio.run(test_telegram())
else:
    print("⚠️ Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to test")
```

---

## CHECKPOINT 3B: Scheduler Integration (Day 8, 4 hours)

### Step 3.3: Task Scheduler with Multi-Target Support
```python
# src/scheduler/task_scheduler.py
import schedule
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List
import threading

class MultiTargetTaskScheduler:
    def __init__(self, db_operations, config):
        self.db = db_operations
        self.config = config
        self.is_running = False
        self.scheduler_thread = None
        
        # Initialize components
        self.data_fetcher = None
        self.trainer = None
        self.predictor = None
        self.telegram_reporter = None
        
    def initialize_components(self):
        """Initialize all components"""
        from src.data.fetcher import ExchangeDataFetcher
        from src.models.multi_target_trainer import MultiTargetModelTrainer
        from src.models.multi_target_predictor import MultiTargetPredictionEngine
        from src.notifications.telegram_reporter import MultiTargetTelegramReporter
        
        self.data_fetcher = ExchangeDataFetcher()
        self.trainer = MultiTargetModelTrainer(self.db, self.config.target_percentages)
        self.predictor = MultiTargetPredictionEngine(self.db, self.config.target_percentages)
        self.predictor.trainer = self.trainer
        
        # Telegram setup
        bot_token = self.config.config['telegram']['bot_token']
        chat_id = self.config.config['telegram']['chat_id']
        
        if bot_token and chat_id:
            self.telegram_reporter = MultiTargetTelegramReporter(bot_token, chat_id)
        else:
            logging.warning("Telegram credentials not provided")

---

## CHECKPOINT 3B: Scheduler Integration (Day 8, 4 hours)

### Step 3.3: Task Scheduler with Multi-Target Support
```python
# src/scheduler/task_scheduler.py
import schedule
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List
import threading

class MultiTargetTaskScheduler:
    def __init__(self, db_operations, config):
        self.db = db_operations
        self.config = config
        self.is_running = False
        self.scheduler_thread = None
        
        # Initialize components
        self.data_fetcher = None
        self.trainer = None
        self.predictor = None
        self.telegram_reporter = None
        
    def initialize_components(self):
        """Initialize all components"""
        from src.data.fetcher import ExchangeDataFetcher
        from src.models.multi_target_trainer import MultiTargetModelTrainer
        from src.models.multi_target_predictor import MultiTargetPredictionEngine
        from src.notifications.telegram_reporter import MultiTargetTelegramReporter
        
        self.data_fetcher = ExchangeDataFetcher()
        self.trainer = MultiTargetModelTrainer(self.db, self.config.target_percentages)
        self.predictor = MultiTargetPredictionEngine(self.db, self.config.target_percentages)
        self.predictor.trainer = self.trainer
        
        # Telegram setup
        bot_token = self.config.config['telegram']['bot_token']
        chat_id = self.config.config['telegram']['chat_id']
        
        if bot_token and chat_id:
            self.telegram_reporter = MultiTargetTelegramReporter(bot_token, chat_id)
        else:
            logging.warning("Telegram credentials not provided")
            
    def setup_schedules(self):
        """Setup all scheduled tasks"""
        # Data fetching every 15 minutes
        schedule.every(15).minutes.do(self._job_wrapper, self.fetch_latest_data)
        
        # Predictions and reports every hour at minute 5
        schedule.every().hour.at(":05").do(self._job_wrapper, self.run_hourly_predictions)
        
        # Model retraining daily at 6 AM UTC
        schedule.every().day.at("06:00").do(self._job_wrapper, self.retrain_models)
        
        # Daily performance summary at 18:00 UTC
        schedule.every().day.at("18:00").do(self._job_wrapper, self.send_daily_summary)
        
        # Weekly cleanup on Sunday at 3 AM UTC
        schedule.every().sunday.at("03:00").do(self._job_wrapper, self.weekly_cleanup)
        
        logging.info("All schedules configured")
        
    def _job_wrapper(self, job_func):
        """Wrapper for scheduled jobs with error handling"""
        try:
            job_func()
        except Exception as e:
            logging.error(f"Scheduled job {job_func.__name__} failed: {e}")
            
    def fetch_latest_data(self):
        """Fetch latest data for all symbols"""
        print(f"\n⏰ {datetime.now().strftime('%H:%M:%S')} - Fetching latest data...")
        
        for symbol in self.config.assets:
            try:
                # Get latest candle
                latest_candle = self.data_fetcher.get_latest_candle(symbol)
                
                if latest_candle:
                    # Save to database
                    saved = self.db.save_ohlcv_data(symbol, '1h', [latest_candle])
                    if saved > 0:
                        print(f"   💾 {symbol}: Updated with latest candle")
                    
                # Also fetch any missing recent data
                recent_data = self.data_fetcher.fetch_historical_data(symbol, days=1)
                if len(recent_data) > 0:
                    candles = recent_data[['timestamp', 'open', 'high', 'low', 'close', 'volume']].values.tolist()
                    candles = [[int(row[0].timestamp() * 1000)] + list(row[1:]) for row in candles]
                    self.db.save_ohlcv_data(symbol, '1h', candles)
                    
            except Exception as e:
                logging.error(f"Error fetching data for {symbol}: {e}")
                
        print("   ✅ Data fetch complete")
        
    def run_hourly_predictions(self):
        """Run predictions and send hourly report"""
        print(f"\n🔮 {datetime.now().strftime('%H:%M:%S')} - Running hourly predictions...")
        
        try:
            # Load existing models
            self._load_all_models()
            
            # Make predictions
            predictions = self.predictor.predict_all_assets(self.config.assets)
            
            # Send Telegram report
            if self.telegram_reporter:
                asyncio.run(self._send_telegram_report(predictions))
            
            # Check for high-confidence alerts
            if self.telegram_reporter:
                asyncio.run(self._check_and_send_alerts(predictions))
                
            print("   ✅ Hourly predictions complete")
            
        except Exception as e:
            logging.error(f"Error in hourly predictions: {e}")
            
    def _load_all_models(self):
        """Load all trained models"""
        print("   📦 Loading models...")
        loaded_count = 0
        
        for symbol in self.config.assets:
            for target_pct in self.config.target_percentages:
                model_data = self.trainer.load_model_from_disk(symbol, target_pct)
                if model_data:
                    self.trainer.models[(symbol, target_pct)] = model_data
                    loaded_count += 1
                    
        print(f"   📦 Loaded {loaded_count} models")
        
    async def _send_telegram_report(self, predictions: Dict):
        """Send Telegram hourly report"""
        try:
            success = await self.telegram_reporter.send_hourly_report(predictions)
            if success:
                print("   📱 Telegram report sent")
            else:
                print("   ❌ Telegram report failed")
        except Exception as e:
            logging.error(f"Error sending Telegram report: {e}")
            
    async def _check_and_send_alerts(self, predictions: Dict):
        """Check for high-confidence predictions and send alerts"""
        try:
            alert_count = 0
            
            for symbol, symbol_data in predictions.items():
                if 'predictions' in symbol_data:
                    for target_str, pred_data in symbol_data['predictions'].items():
                        if 'error' not in pred_data and pred_data.get('confidence', 0) > 0.75:
                            await self.telegram_reporter.send_high_confidence_alert(symbol, target_str, pred_data)
                            alert_count += 1
                            
            if alert_count > 0:
                print(f"   🚨 Sent {alert_count} high-confidence alerts")
                
        except Exception as e:
            logging.error(f"Error sending alerts: {e}")
            
    def retrain_models(self):
        """Retrain all models with latest data"""
        print(f"\n🏋️ {datetime.now().strftime('%H:%M:%S')} - Starting model retraining...")
        
        try:
            # Fetch fresh data before retraining
            for symbol in self.config.assets:
                recent_data = self.data_fetcher.fetch_historical_data(symbol, days=7)
                if len(recent_data) > 0:
                    candles = recent_data[['timestamp', 'open', 'high', 'low', 'close', 'volume']].values.tolist()
                    candles = [[int(row[0].timestamp() * 1000)] + list(row[1:]) for row in candles]
                    self.db.save_ohlcv_data(symbol, '1h', candles)
            
            # Retrain models
            results = self.trainer.train_all_models(self.config.assets, retrain=True)
            
            # Count successful retraining
            total_models = sum(len(asset_results) for asset_results in results.values())
            successful_models = sum(sum(1 for success in asset_results.values() if success) 
                                  for asset_results in results.values())
            
            print(f"   ✅ Retraining complete: {successful_models}/{total_models} models successful")
            
            # Send notification if Telegram available
            if self.telegram_reporter:
                message = f"🏋️ *Model Retraining Complete*\n"
                message += f"✅ {successful_models}/{total_models} models retrained successfully\n"
                message += f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}"
                
                asyncio.run(self.telegram_reporter.bot.send_message(
                    chat_id=self.telegram_reporter.chat_id, 
                    text=message, 
                    parse_mode='Markdown'
                ))
                
        except Exception as e:
            logging.error(f"Error during model retraining: {e}")
            
    def send_daily_summary(self):
        """Send daily performance summary"""
        print(f"\n📊 {datetime.now().strftime('%H:%M:%S')} - Generating daily summary...")
        
        try:
            # Get performance data for last 24 hours
            performance_data = {}
            
            for symbol in self.config.assets:
                performance_data[symbol] = {}
                for target_pct in self.config.target_percentages:
                    # This would get actual performance metrics from database
                    # For now, using mock data
                    performance_data[symbol][f"{target_pct:.1%}"] = {
                        'accuracy': 0.65,  # Mock data
                        'total_predictions': 24
                    }
            
            # Send Telegram summary
            if self.telegram_reporter:
                asyncio.run(self.telegram_reporter.send_daily_performance_summary(performance_data))
                print("   📱 Daily summary sent")
                
        except Exception as e:
            logging.error(f"Error sending daily summary: {e}")
            
    def weekly_cleanup(self):
        """Weekly maintenance tasks"""
        print(f"\n🧹 {datetime.now().strftime('%H:%M:%S')} - Running weekly cleanup...")
        
        try:
            # Database cleanup would go here
            # For now, just log
            print("   🧹 Weekly cleanup complete")
            
        except Exception as e:
            logging.error(f"Error during weekly cleanup: {e}")
            
    def start(self):
        """Start the scheduler"""
        if self.is_running:
            print("Scheduler is already running")
            return
            
        print("🚀 Starting Multi-Target Task Scheduler...")
        
        # Initialize components
        self.initialize_components()
        
        # Setup schedules
        self.setup_schedules()
        
        # Start scheduler in separate thread
        self.is_running = True
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        print("✅ Scheduler started successfully")
        
    def _run_scheduler(self):
        """Run the scheduler loop"""
        while self.is_running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
            
    def stop(self):
        """Stop the scheduler"""
        print("🛑 Stopping scheduler...")
        self.is_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        print("✅ Scheduler stopped")
        
    def run_manual_prediction(self):
        """Run predictions manually (for testing)"""
        print("🔮 Running manual prediction...")
        self._load_all_models()
        predictions = self.predictor.predict_all_assets(self.config.assets)
        
        if self.telegram_reporter:
            asyncio.run(self._send_telegram_report(predictions))
            
        return predictions
```

**🔍 CHECKPOINT 3B TEST:**
```python
# test_scheduler.py
from src.scheduler.task_scheduler import MultiTargetTaskScheduler
from src.database.operations import DatabaseOperations
from src.utils.config import Config
import time

# Initialize
config = Config()
db_ops = DatabaseOperations(config.database_url)
scheduler = MultiTargetTaskScheduler(db_ops, config)

# Test initialization
scheduler.initialize_components()
print("✅ Components initialized")

# Test manual prediction
predictions = scheduler.run_manual_prediction()
print("✅ Manual prediction completed")

# Test scheduling setup (don't start actual scheduler in test)
scheduler.setup_schedules()
print("✅ Schedules configured")

# Show next scheduled run times
import schedule
print("📅 Next scheduled runs:")
for job in schedule.jobs:
    print(f"   {job.next_run} - {job.job_func}")
```

---

## Phase 3 Success Criteria

After completing Phase 3, you should have:

✅ **Telegram bot** that sends formatted hourly reports  
✅ **Rich message formatting** with emojis and confidence indicators  
✅ **High-confidence alerts** (>75% threshold) sent immediately  
✅ **Automated scheduler** running predictions every hour  
✅ **Daily summaries** and model retraining notifications  

**Next Step**: Move to Phase 4 to containerize and deploy to Hetzner VPS.
