import logging
import signal
import sys
import os
from flask import Flask, jsonify
import threading
import time
from datetime import datetime

# Configure logging
log_handlers = [logging.StreamHandler(sys.stdout)]

# Try to add file handler
log_file_path = None
if os.path.exists('/app/logs') and os.access('/app/logs', os.W_OK):
    log_file_path = '/app/logs/app.log'
elif os.path.exists('logs') and os.access('logs', os.W_OK):
    log_file_path = 'logs/app.log'
else:
    # Try to create logs directory
    try:
        os.makedirs('logs', exist_ok=True)
        log_file_path = 'logs/app.log'
    except:
        print("Warning: Unable to create logs directory, logging to console only")

if log_file_path:
    try:
        log_handlers.append(logging.FileHandler(log_file_path))
    except PermissionError:
        print(f"Warning: Unable to write to {log_file_path}, logging to console only")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=log_handlers
)

logger = logging.getLogger(__name__)

class CryptoMLApplication:
    def __init__(self):
        self.scheduler = None
        self.flask_app = None
        self.is_running = False
        
    def initialize(self):
        """Initialize application components"""
        try:
            logger.info("🚀 Initializing Crypto ML Multi-Target Predictor")
            
            # Load configuration
            from src.utils.config import config
            self.config = config
            logger.info("✅ Configuration loaded")
            
            # Enable database error logging after initial setup
            from src.utils.simple_db_logger import add_db_error_handler
            add_db_error_handler()
            logger.info("✅ Database error logging enabled")
            
            # Initialize database
            from src.database.operations import db_ops
            self.db_ops = db_ops
            logger.info("✅ Database connection established")
            
            # Initialize database tables
            from src.database.init_db import init_database
            logger.info("🗄️ Initializing database tables...")
            table_status = init_database(self.config.database_url)
            logger.info("✅ Database tables initialized")
            
            # Initialize scheduler
            from src.scheduler.task_scheduler import MultiTargetTaskScheduler
            self.scheduler = MultiTargetTaskScheduler(self.db_ops, self.config)
            logger.info("✅ Task scheduler initialized")
            
            # Setup Flask health check endpoint
            self.setup_health_check()
            logger.info("✅ Health check endpoint configured")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            return False
            
    def setup_health_check(self):
        """Setup Flask health check endpoint"""
        self.flask_app = Flask(__name__)
        
        @self.flask_app.route('/health')
        def health_check():
            try:
                # Check database connection
                db_status = self.db_ops.db.test_connection() if self.db_ops else False
                
                # Check scheduler status
                scheduler_status = self.scheduler.is_running if self.scheduler else False
                
                # Check if we have recent predictions
                recent_predictions = self._check_recent_predictions()
                
                status = {
                    'status': 'healthy' if (db_status and scheduler_status) else 'unhealthy',
                    'timestamp': datetime.now().isoformat(),
                    'components': {
                        'database': 'up' if db_status else 'down',
                        'scheduler': 'running' if scheduler_status else 'stopped',
                        'predictions': 'current' if recent_predictions else 'stale'
                    },
                    'uptime_seconds': int(time.time() - self.start_time) if hasattr(self, 'start_time') else 0
                }
                
                return jsonify(status), 200 if status['status'] == 'healthy' else 503
                
            except Exception as e:
                return jsonify({
                    'status': 'error',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
                
    def _check_recent_predictions(self):
        """Check if we have predictions from the last 2 hours"""
        try:
            from datetime import timedelta
            cutoff_time = datetime.now() - timedelta(hours=2)
            
            # Check for recent predictions in database
            with self.db_ops.db.get_session() as session:
                from sqlalchemy import text
                result = session.execute(
                    text("""
                        SELECT COUNT(*) as count
                        FROM predictions
                        WHERE created_at > :cutoff_time
                    """),
                    {'cutoff_time': cutoff_time}
                )
                row = result.fetchone()
                
            return row[0] > 0 if row else False
        except:
            return False
                
    def start_health_server(self):
        """Start Flask health check server"""
        self.flask_app.run(host='0.0.0.0', port=8000, debug=False, use_reloader=False)
        
    def run(self):
        """Run the application"""
        if not self.initialize():
            logger.error("❌ Failed to initialize application")
            sys.exit(1)
            
        try:
            self.start_time = time.time()
            self.is_running = True
            
            # Start health check server in separate thread
            health_thread = threading.Thread(target=self.start_health_server, daemon=True)
            health_thread.start()
            logger.info("✅ Health check server started on port 8000")
            
            # Initial data fetch and model training
            logger.info("🔄 Running initial setup...")
            self.run_initial_setup()
            
            # Start scheduler
            logger.info("⏰ Starting task scheduler...")
            self.scheduler.start()
            # Note: scheduler.start() already runs initial predictions
            
            logger.info("🎉 Application started successfully!")
            logger.info("📱 Hourly reports will be sent to Telegram")
            logger.info("🔄 Models will retrain daily at 06:00 UTC")
            
            # Keep main thread alive
            while self.is_running:
                time.sleep(60)
                
        except KeyboardInterrupt:
            logger.info("🛑 Shutdown requested by user")
        except Exception as e:
            logger.error(f"❌ Application error: {e}")
        finally:
            self.shutdown()
            
    def run_initial_setup(self):
        """Run initial data fetch and model training with recovery"""
        try:
            # Initialize data recovery manager
            from src.data.recovery import recovery_manager
            
            # FIRST: Create futures table before any data operations
            logger.info("📊 Creating database tables...")
            try:
                self.db_ops.create_futures_table()
                logger.info("✅ Futures table ready")
            except Exception as e:
                logger.warning(f"⚠️ Futures table creation: {e}")
                # Continue anyway - table might already exist
            
            # Skip verbose gap checking - the ensure_all_symbols_coverage will handle it
            # logger.info("🔍 Checking data integrity...")
            # recovery_results = recovery_manager.check_and_recover_all_symbols(
            #     hours_back=168  # Check last week
            # )
            # 
            # for symbol, filled in recovery_results.items():
            #     if filled > 0:
            #         logger.info(f"✅ Recovered {filled} missing candles for {symbol}")
                    
            # SECOND: Ensure spot data coverage
            logger.info("📊 Ensuring spot data coverage (up to 3 years)...")
            coverage_results = recovery_manager.ensure_all_symbols_coverage()
            
            for symbol, count in coverage_results.items():
                if count >= 1000:
                    logger.info(f"✅ {symbol}: {count} candles available")
                else:
                    logger.warning(f"⚠️ {symbol}: Only {count} candles available")
            
            # THIRD: Initialize futures data
            logger.info("📊 Initializing futures data...")
            try:
                # Ensure futures data coverage - try to match spot data coverage
                futures_results = recovery_manager.ensure_all_symbols_futures_coverage()
                
                for symbol, success in futures_results.items():
                    if success:
                        logger.info(f"✅ {symbol}: Futures data initialized")
                    else:
                        logger.warning(f"⚠️ {symbol}: Failed to initialize futures data")
                        
                # Fetch latest futures data
                logger.info("📈 Fetching latest futures data...")
                latest_futures = recovery_manager.fetch_latest_futures_data()
                
                for symbol, data in latest_futures.items():
                    if data:
                        logger.info(f"💾 {symbol}: OI={data.get('open_interest', 'N/A')}, FR={data.get('funding_rate', 'N/A')}")
                    else:
                        logger.warning(f"⚠️ {symbol}: No futures data available")
                        
            except Exception as e:
                logger.error(f"❌ Futures data initialization failed: {e}")
                logger.info("📌 Continuing without futures data...")
                
            # Train initial models if not exist
            logger.info("🏋️ Training initial models...")
            from src.models.multi_target_trainer import MultiTargetModelTrainer
            trainer = MultiTargetModelTrainer(self.config.target_percentages)
            
            # Always retrain models on startup to ensure they use latest data
            # and the corrected target labeling logic
            logger.info("🏋️ Training models with latest data...")
            result = trainer.train_all_models(retrain=True)
            print("✅ Model training completed, moving to tuning check...")

            # 🧪 Hyperparameter tuning across all configured model types (may take a long time)
            print("🔍 Checking if tuning is enabled...")
            try:
                tuning_enabled = self.config.get_nested("ml.tuning.enabled", True)
                print(f"🔍 Tuning enabled config value: {tuning_enabled}")
                if tuning_enabled:
                    from src.tuning.optuna_tuner import OptunaHyperparameterTuner
                    tuner = OptunaHyperparameterTuner()
                    model_types = self.config.get_nested("ml.tuning.models", ["rf", "xgb", "lgb"])
                    n_trials = int(self.config.get_nested("ml.tuning.n_trials", 50))

                    print("🧪 Starting hyperparameter tuning for all symbols/targets")
                    for symbol in self.config.assets:
                        for target_pct in self.config.target_percentages:
                            for model_type in model_types:
                                print(f"🧪 Tuning {symbol} {target_pct:.1%} [{model_type}] with {n_trials} trials")
                                try:
                                    tuner.run(
                                        symbol=symbol,
                                        target_pct=target_pct,
                                        model_type=model_type,
                                        n_trials_override=n_trials,
                                    )
                                except Exception as e:
                                    print(f"❌ Tuning failed for {symbol} {target_pct:.1%} [{model_type}]: {e}")
                    print("✅ Hyperparameter tuning complete")
                else:
                    print("ℹ️ Hyperparameter tuning disabled via config")
            except Exception as e:
                print(f"❌ Failed during hyperparameter tuning orchestration: {e}")
            
        except Exception as e:
            logger.error(f"❌ Initial setup failed: {e}")
            # Continue anyway - scheduler will retry
            
    def shutdown(self):
        """Graceful shutdown"""
        logger.info("🛑 Shutting down application...")
        self.is_running = False
        
        if self.scheduler:
            self.scheduler.stop()
            
        logger.info("✅ Application shutdown complete")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"🛑 Received signal {signum}")
    sys.exit(0)

if __name__ == "__main__":
    # Setup signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create and run application
    app = CryptoMLApplication()
    app.run()
