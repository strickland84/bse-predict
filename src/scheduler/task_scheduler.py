import schedule
import time
import logging
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List
import threading
import psutil
import os

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
        from src.data.recovery import recovery_manager
        from src.tracking.outcome_monitor import outcome_monitor
        
        self.data_fetcher = ExchangeDataFetcher()
        self.trainer = MultiTargetModelTrainer(self.config.target_percentages)
        self.predictor = MultiTargetPredictionEngine(self.config.target_percentages)
        self.predictor.trainer = self.trainer
        self.outcome_monitor = outcome_monitor
        
        # Telegram setup
        bot_token = self.config.config['telegram']['bot_token']
        chat_id = self.config.config['telegram']['chat_id']
        
        if bot_token and chat_id:
            self.telegram_reporter = MultiTargetTelegramReporter(bot_token, chat_id)
            logging.info("Telegram reporter initialized")
        else:
            logging.warning("Telegram credentials not provided")
            
        # Check if data initialization was already done by main.py
        min_required_candles = 1000
        needs_data_init = False
        
        for symbol in self.config.assets:
            count = self.db.get_candle_count(symbol)
            if count < min_required_candles:
                needs_data_init = True
                break
        
        if not needs_data_init:
            # Data already initialized, just show status
            print("\n✅ Data already initialized by main application")
            print("📈 Current data status:")
            for symbol in self.config.assets:
                count = self.db.get_candle_count(symbol)
                latest = self.db.get_latest_candles(symbol, limit=1)
                if not latest.empty:
                    latest_time = latest.iloc[0]['timestamp']
                    hours_old = (datetime.now(timezone.utc) - latest_time).total_seconds() / 3600
                    print(f"   {symbol}: {count} candles (latest: {hours_old:.1f} hours old)")
        else:
            # Need to initialize data
            print("\n📊 Ensuring data availability...")
            print("   This may take a few minutes on first run...")
            
            # Check current data status first
            print("\n📈 Current spot data status:")
            for symbol in self.config.assets:
                count = self.db.get_candle_count(symbol)
                if count > 0:
                    latest = self.db.get_latest_candles(symbol, limit=1)
                    if not latest.empty:
                        latest_time = latest.iloc[0]['timestamp']
                        hours_old = (datetime.now(timezone.utc) - latest_time).total_seconds() / 3600
                        print(f"   {symbol}: {count} candles (latest: {hours_old:.1f} hours old)")
                    else:
                        print(f"   {symbol}: {count} candles (no recent data)")
                else:
                    print(f"   {symbol}: No data")
            
            # Ensure we have enough candles for each symbol
            print("\n📥 Fetching/updating spot data if needed...")
            training_limit = self.config.get_nested('ml.training_data_limit', 4000)
            data_results = recovery_manager.ensure_all_symbols_coverage(min_candles=training_limit)
            
            print("\n✅ Spot data ready:")
            for symbol, count in data_results.items():
                print(f"   {symbol}: {count} candles available")
            
        # Only initialize futures if we did spot data initialization
        if needs_data_init:
            print("\n📊 Ensuring futures data availability...")
            try:
                # Create futures table if it doesn't exist
                self.db.create_futures_table()
                logging.info("Futures table ready")
                
                # Ensure futures data coverage - try to match spot data coverage
                futures_results = recovery_manager.ensure_all_symbols_futures_coverage()
                
                print("\n✅ Futures data ready:")
                for symbol, success in futures_results.items():
                    status = "✅" if success else "❌"
                    print(f"   {status} {symbol}: Futures data initialized")
                    
            except Exception as e:
                logging.error(f"Error initializing futures data: {e}")
                print(f"\n⚠️ Warning: Could not initialize futures data: {e}")
            
        # Check if we need to train models
        self._ensure_models_trained()
            
    def setup_schedules(self):
        """Setup all scheduled tasks"""
        # Data fetching every 15 minutes
        schedule.every(15).minutes.do(self._job_wrapper, self.fetch_latest_data)
        
        # Predictions and reports every hour at minute 5
        schedule.every().hour.at(":05").do(self._job_wrapper, self.run_hourly_predictions)
        
        # Model retraining daily at 6:10 AM UTC
        schedule.every().day.at("06:10").do(self._job_wrapper, self.retrain_models)
        
        # Daily performance summary at 18:00 UTC
        schedule.every().day.at("18:00").do(self._job_wrapper, self.send_daily_summary)
        
        # Weekly cleanup on Sunday at 3 AM UTC
        schedule.every().sunday.at("03:00").do(self._job_wrapper, self.weekly_cleanup)
        
        # System health checks (every 30 minutes)
        schedule.every(30).minutes.do(self._job_wrapper, self.record_system_health)
        
        logging.info("All schedules configured")
        
    def _job_wrapper(self, job_func):
        """Wrapper for scheduled jobs with error handling"""
        try:
            job_func()
        except Exception as e:
            logging.error(f"Scheduled job {job_func.__name__} failed: {e}")
            
    def fetch_latest_data(self):
        """Fetch latest spot and futures data for all symbols"""
        print(f"\n⏰ {datetime.now().strftime('%H:%M:%S')} - Fetching latest data...")
        
        # Fetch spot data
        for symbol in self.config.assets:
            try:
                # Get latest candle
                latest_candle = self.data_fetcher.get_latest_candle(symbol)
                
                if latest_candle:
                    # Save to database
                    saved = self.db.save_ohlcv_data(symbol, '1h', [latest_candle])
                    if saved > 0:
                        print(f"   💾 {symbol}: Updated with latest spot candle")
                    
                # Also fetch any missing recent data
                recent_data = self.data_fetcher.fetch_historical_data(symbol, days=1)
                if len(recent_data) > 0:
                    candles = recent_data[['timestamp', 'open', 'high', 'low', 'close', 'volume']].values.tolist()
                    candles = [[int(row[0].timestamp() * 1000)] + list(row[1:]) for row in candles]
                    self.db.save_ohlcv_data(symbol, '1h', candles)
                    
            except Exception as e:
                logging.error(f"Error fetching spot data for {symbol}: {e}")
        
        # Fetch futures data
        print("   📊 Fetching latest futures data...")
        try:
            from src.data.recovery import recovery_manager
            futures_results = recovery_manager.fetch_latest_futures_data(self.config.assets)
            
            for symbol, data in futures_results.items():
                if data:
                    print(f"   💾 {symbol}: Updated futures data (OI: {data.get('open_interest', 'N/A')}, FR: {data.get('funding_rate', 'N/A')})")
                else:
                    print(f"   ⚠️ {symbol}: No futures data available")
                    
        except Exception as e:
            logging.error(f"Error fetching futures data: {e}")
            print(f"   ❌ Futures data fetch failed: {e}")
                
        print("   ✅ Data fetch complete")
        
    def run_hourly_predictions(self):
        """Run predictions and send hourly report"""
        print(f"\n🔮 {datetime.now().strftime('%H:%M:%S')} - Running hourly predictions...")
        
        try:
            # Track prediction timing
            start_time = time.time()
            
            # Load existing models
            self._load_all_models()
            
            # Make predictions
            predictions = self.predictor.predict_all_assets(self.config.assets)
            
            # Calculate prediction latency
            prediction_latency_ms = int((time.time() - start_time) * 1000)
            
            # Send Telegram report
            if self.telegram_reporter:
                try:
                    success = self.telegram_reporter.send_hourly_report(predictions, trigger_type="scheduled")
                    if success:
                        print("   📱 Telegram report sent")
                    
                    # Check for high confidence alerts
                    self._check_and_send_alerts(predictions)
                except Exception as e:
                    logging.error(f"Telegram error: {e}")
            
            # Check prediction outcomes
            self.check_prediction_outcomes()
            
            # Record system health
            from src.monitoring.system_health import system_health_monitor
            system_health_monitor.record_health_check(prediction_latency_ms)
                
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
        
    def _ensure_models_trained(self):
        """Ensure models are trained, train if necessary"""
        print("\n🤖 Checking model availability...")
        
        # Try to load existing models
        loaded_count = 0
        missing_models = []
        
        for symbol in self.config.assets:
            for target_pct in self.config.target_percentages:
                model_data = self.trainer.load_model_from_disk(symbol, target_pct)
                if model_data:
                    self.trainer.models[(symbol, target_pct)] = model_data
                    loaded_count += 1
                else:
                    missing_models.append((symbol, target_pct))
                    
        total_models = len(self.config.assets) * len(self.config.target_percentages)
        print(f"   Found {loaded_count}/{total_models} existing models")
        
        if missing_models:
            print(f"\n🏋️ Training {len(missing_models)} missing models...")
            print("   This will take a few minutes on first run...")
            
            # Train all models with retrain=True to ensure fresh models
            results = self.trainer.train_all_models(self.config.assets, retrain=True)
            
            # Count successful training
            successful = sum(sum(1 for success in asset_results.values() if success) 
                           for asset_results in results.values())
            
            print(f"   ✅ Trained {successful}/{len(missing_models)} models successfully")
            
            # Reload all models to ensure predictor has them
            self._load_all_models()
        else:
            print("   ✅ All models already trained and loaded")
        
    def _check_and_send_alerts(self, predictions: Dict):
        """Check for high-confidence predictions and send alerts"""
        try:
            alert_count = 0
            
            for symbol, symbol_data in predictions.items():
                if 'predictions' in symbol_data:
                    for target_str, pred_data in symbol_data['predictions'].items():
                        if 'error' not in pred_data and pred_data.get('confidence', 0) > 0.75:
                            if self.telegram_reporter.send_high_confidence_alert(symbol, target_str, pred_data):
                                alert_count += 1
                            
            if alert_count > 0:
                print(f"   🚨 Sent {alert_count} high-confidence alerts")
                
        except Exception as e:
            logging.error(f"Error sending alerts: {e}")
            
    def retrain_models(self):
        """Retrain all models with latest data"""
        print(f"\n🏋️ {datetime.now().strftime('%H:%M:%S')} - Starting model retraining...")
        
        try:
            # Fetch fresh spot data before retraining
            print("   📥 Updating spot data...")
            for symbol in self.config.assets:
                recent_data = self.data_fetcher.fetch_historical_data(symbol, days=7)
                if len(recent_data) > 0:
                    candles = recent_data[['timestamp', 'open', 'high', 'low', 'close', 'volume']].values.tolist()
                    candles = [[int(row[0].timestamp() * 1000)] + list(row[1:]) for row in candles]
                    saved_count = self.db.save_ohlcv_data(symbol, '1h', candles)
                    if saved_count > 0:
                        logging.info(f"Updated {saved_count} spot candles for {symbol}")
            
            # Fetch fresh futures data before retraining
            print("   📊 Updating futures data...")
            try:
                from src.data.recovery import recovery_manager
                # Ensure at least 7 days of futures data
                futures_results = recovery_manager.ensure_all_symbols_futures_coverage(min_hours=168)
                
                # Also fetch latest futures data
                latest_futures = recovery_manager.fetch_latest_futures_data(self.config.assets)
                
                for symbol, data in latest_futures.items():
                    if data:
                        logging.info(f"Updated futures data for {symbol}: OI={data.get('open_interest', 'N/A')}")
                        
            except Exception as e:
                logging.error(f"Error updating futures data: {e}")
                print(f"   ⚠️ Warning: Could not update futures data: {e}")
            
            # Retrain models
            results = self.trainer.train_all_models(self.config.assets, retrain=True)
            
            # Count successful retraining
            total_models = sum(len(asset_results) for asset_results in results.values())
            successful_models = sum(sum(1 for success in asset_results.values() if success) 
                                  for asset_results in results.values())
            
            print(f"   ✅ Retraining complete: {successful_models}/{total_models} models successful")
            
            # 🧪 Hyperparameter tuning across all configured model types (may take a long time)
            try:
                if self.config.get_nested("ml.tuning.enabled", True):
                    from src.tuning.optuna_tuner import OptunaHyperparameterTuner
                    tuner = OptunaHyperparameterTuner()
                    model_types = self.config.get_nested("ml.tuning.models", ["rf", "xgb", "lgb"])
                    n_trials = int(self.config.get_nested("ml.tuning.n_trials", 50))

                    logging.info("🧪 Starting hyperparameter tuning for all symbols/targets (post-retrain)")
                    for symbol in self.config.assets:
                        for target_pct in self.config.target_percentages:
                            for model_type in model_types:
                                logging.info(f"🧪 Tuning {symbol} {target_pct:.1%} [{model_type}] with {n_trials} trials")
                                try:
                                    tuner.run(
                                        symbol=symbol,
                                        target_pct=target_pct,
                                        model_type=model_type,
                                        n_trials_override=n_trials,
                                    )
                                except Exception as e:
                                    logging.error(f"Optuna tuning failed for {symbol} {target_pct:.1%} [{model_type}]: {e}")
                    logging.info("✅ Hyperparameter tuning complete")
                else:
                    logging.info("ℹ️ Hyperparameter tuning disabled via config")
            except Exception as e:
                logging.error(f"❌ Hyperparameter tuning orchestration failed: {e}")
            
            # Send notification if Telegram available
            if self.telegram_reporter:
                message = f"🏋️ *DAILY MODEL RETRAINING COMPLETE*\n"
                message += f"🔄 Scheduled retraining at 06:00 UTC\n"
                message += "─" * 30 + "\n\n"
                message += f"✅ {successful_models}/{total_models} models retrained successfully\n"
                message += f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}\n"
                message += f"\n📌 _Models are retrained daily with fresh market data_"
                
                try:
                    self.telegram_reporter.send_message(message)
                except Exception as e:
                    logging.error(f"Error sending retraining notification: {e}")
                
        except Exception as e:
            logging.error(f"Error during model retraining: {e}")
            
    def send_daily_summary(self):
        """Send daily performance summary with real prediction outcomes"""
        print(f"\n📊 {datetime.now().strftime('%H:%M:%S')} - Generating daily summary...")
        
        try:
            # Calculate and save model performance for yesterday
            from src.analytics.model_performance import model_performance_tracker
            model_performance_tracker.update_daily_performance(days_back=1)
            
            # Get real performance data from prediction tracker
            from src.database.prediction_tracker import prediction_tracker
            
            performance_data = {}
            
            for symbol in self.config.assets:
                performance_data[symbol] = {}
                
                for target_pct in self.config.target_percentages:
                    # Get actual performance metrics for the last 24 hours
                    perf_df = prediction_tracker.get_prediction_performance(
                        symbol=symbol,
                        target_pct=target_pct,
                        days_back=1  # Last 24 hours
                    )
                    
                    if not perf_df.empty and len(perf_df) > 0:
                        row = perf_df.iloc[0]
                        performance_data[symbol][f"{target_pct:.1%}"] = {
                            'accuracy': float(row['accuracy']) if pd.notna(row['accuracy']) else 0,
                            'total_predictions': int(row['total_predictions']),
                            'tracked_predictions': int(row['tracked_predictions']),
                            'avg_confidence': float(row['avg_confidence']) if pd.notna(row['avg_confidence']) else 0,
                            'avg_time_to_target': float(row['avg_time_to_target']) if pd.notna(row['avg_time_to_target']) else 0
                        }
                    else:
                        # No data for this symbol/target
                        performance_data[symbol][f"{target_pct:.1%}"] = {
                            'accuracy': 0,
                            'total_predictions': 0,
                            'tracked_predictions': 0
                        }
            
            # Send enhanced Telegram summary
            if self.telegram_reporter:
                try:
                    # Send standard performance summary
                    if self.telegram_reporter.send_daily_performance_summary(performance_data):
                        print("   📱 Daily summary sent")
                    
                    # Also send detailed analytics report if we have outcomes
                    total_tracked = sum(
                        data.get('tracked_predictions', 0)
                        for symbol_data in performance_data.values()
                        for data in symbol_data.values()
                    )
                    
                    if total_tracked > 0:
                        from src.analytics.prediction_analytics import prediction_analytics
                        analytics_report = prediction_analytics.generate_analytics_report()
                        self.telegram_reporter.send_message(analytics_report)
                        print("   📊 Analytics report sent")
                        
                except Exception as e:
                    logging.error(f"Error sending daily summary: {e}")
                
        except Exception as e:
            logging.error(f"Error sending daily summary: {e}")
            
    def _format_outcome_monitoring_message(self, stats: Dict) -> str:
        """Format outcome monitoring statistics for Telegram.
        
        Args:
            stats: Dictionary with outcome statistics
            
        Returns:
            Formatted message string
        """
        from datetime import datetime
        
        # Calculate completion rate
        if stats['checked'] > 0:
            completion_rate = (stats['completed'] / stats['checked']) * 100
        else:
            completion_rate = 0
            
        # Build the message
        lines = [
            "🔍 **OUTCOME MONITORING UPDATE**",
            f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}",
            "=" * 30,
            "",
            f"**Predictions Monitored:** {stats['checked']}",
        ]
        
        if stats['checked'] > 0:
            lines.extend([
                "",
                "**Status Breakdown:**",
                f"✅ Correct: {stats['hit_target']}",
                f"❌ Incorrect: {stats['missed_target']}",
                f"⏱️ Expired: {stats.get('expired', 0)}",
                f"⏳ Still Pending: {stats['still_pending']}",
                f"📊 Completed: {stats['completed']} ({completion_rate:.1f}%)",
            ])
            
            # Add accuracy if any completed with targets hit
            targets_hit = stats['hit_target'] + stats['missed_target']
            if targets_hit > 0:
                accuracy = (stats['hit_target'] / targets_hit) * 100
                lines.extend([
                    "",
                    f"**Accuracy:** {accuracy:.1f}% ({stats['hit_target']}/{targets_hit})",
                ])
                
                # Add emoji based on accuracy
                if accuracy >= 70:
                    lines.append("🎯 Excellent performance!")
                elif accuracy >= 50:
                    lines.append("📈 Good performance")
                else:
                    lines.append("📉 Below expectations")
            
            # Add completed prediction details if any
            if stats['completed'] > 0 and 'completed_details' in stats:
                lines.extend(["", "**Recently Completed:**"])
                for detail in stats['completed_details'][:5]:  # Show max 5
                    symbol = detail['symbol']
                    target = detail['target_pct']
                    predicted = detail['predicted']
                    result = detail['result']
                    
                    # Choose emoji based on result
                    if result == 'CORRECT':
                        emoji = "✅"
                    elif result == 'INCORRECT':
                        emoji = "❌"
                    else:  # EXPIRED
                        emoji = "⏱️"
                    
                    # Format the detail line
                    detail_line = f"{emoji} {symbol} ±{target:.1%} {predicted}"
                    if detail['time_to_target'] and result != 'EXPIRED':
                        detail_line += f" ({detail['time_to_target']:.1f}h)"
                    
                    lines.append(detail_line)
                
                if len(stats['completed_details']) > 5:
                    lines.append(f"... and {len(stats['completed_details']) - 5} more")
        else:
            lines.append("ℹ️ No predictions to monitor at this time")
            
        lines.extend([
            "",
            "🔄 Next check in 1 hour"
        ])
        
        return "\n".join(lines)
    
    def check_prediction_outcomes(self):
        """Check and update prediction outcomes"""
        print("   📈 Checking prediction outcomes...")
        
        try:
            stats = self.outcome_monitor.check_prediction_outcomes()
            
            if stats['checked'] > 0:
                print(f"   📊 Checked {stats['checked']} predictions: "
                      f"{stats['hit_target']} hit, {stats['missed_target']} missed, "
                      f"{stats.get('expired', 0)} expired, "
                      f"{stats['still_pending']} pending")
                
                # Always send outcome monitoring status to Telegram
                if self.telegram_reporter and (stats['completed'] > 0 or stats['checked'] > 0):
                    try:
                        # Create outcome monitoring status message
                        monitoring_msg = self._format_outcome_monitoring_message(stats)
                        self.telegram_reporter.send_message(monitoring_msg)
                        
                        # Send detailed performance report if significant completions
                        if stats['completed'] >= 5:
                            report = self.outcome_monitor.generate_performance_report()
                            self.telegram_reporter.send_message(report)
                    except Exception as e:
                        logging.error(f"Error sending outcome report: {e}")
                        
        except Exception as e:
            logging.error(f"Error checking prediction outcomes: {e}")
    
    def weekly_cleanup(self):
        """Weekly maintenance tasks"""
        print(f"\n🧹 {datetime.now().strftime('%H:%M:%S')} - Running weekly cleanup...")
        
        try:
            # Database cleanup
            from src.database.prediction_tracker import prediction_tracker
            deleted = prediction_tracker.cleanup_old_predictions(days_to_keep=30)
            print(f"   🗑️ Cleaned up {deleted} old predictions")
            
            print("   🧹 Weekly cleanup complete")
            
        except Exception as e:
            logging.error(f"Error during weekly cleanup: {e}")
    
    def record_system_health(self):
        """Record system health metrics"""
        try:
            from src.monitoring.system_health import system_health_monitor
            system_health_monitor.record_health_check()
            logging.debug("System health check recorded")
        except Exception as e:
            logging.error(f"Error recording system health: {e}")
            
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
        
        # Run initial prediction
        print("\n🔮 Running initial predictions...")
        try:
            self.run_manual_prediction()
        except Exception as e:
            logging.error(f"Error in initial predictions: {e}")
            print(f"   ❌ Initial prediction failed: {e}")
        
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
            try:
                success = self.telegram_reporter.send_hourly_report(predictions, trigger_type="startup")
                if success:
                    print("   📱 Telegram report sent")
                else:
                    print("   ❌ Telegram report failed")
            except Exception as e:
                logging.error(f"Error in manual prediction Telegram report: {e}")
            
        return predictions
        
    def get_system_status(self) -> Dict:
        """Get current system status"""
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'active_models': len(self.trainer.models) if self.trainer else 0,
            'last_data_update': datetime.now().isoformat(),
            'scheduler_running': self.is_running
        }
