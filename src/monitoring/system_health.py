"""System health monitoring module."""
import psutil
import time
from datetime import datetime
from typing import Dict, Optional

from src.database.operations import db_ops
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SystemHealthMonitor:
    """Monitor and record system health metrics."""
    
    def __init__(self):
        self.start_time = time.time()
        
    def collect_metrics(self, prediction_latency_ms: Optional[int] = None) -> Dict:
        """Collect current system health metrics.
        
        Args:
            prediction_latency_ms: Time taken for last prediction batch (optional)
            
        Returns:
            Dictionary of health metrics
        """
        try:
            # CPU and memory metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Data freshness - check latest OHLCV data
            data_freshness_minutes = self._get_data_freshness()
            
            # Active models count
            active_models = self._count_active_models()
            
            metrics = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'disk_percent': disk.percent,
                'prediction_latency_ms': prediction_latency_ms,
                'data_freshness_minutes': data_freshness_minutes,
                'active_models': active_models
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {}
    
    def _get_data_freshness(self) -> int:
        """Get minutes since last data update."""
        try:
            query = """
            SELECT EXTRACT(EPOCH FROM (NOW() - MAX(timestamp))) / 60 as minutes_old
            FROM ohlcv_data
            WHERE timeframe = '1h'
            """
            with db_ops.db.get_session() as session:
                from sqlalchemy import text
                result = session.execute(text(query))
                row = result.fetchone()
                if row and row[0]:
                    return int(row[0])
                return -1
        except Exception as e:
            logger.error(f"Error checking data freshness: {e}")
            return -1
    
    def _count_active_models(self) -> int:
        """Count number of trained models."""
        try:
            from pathlib import Path
            models_dir = Path("models")
            if models_dir.exists():
                return len(list(models_dir.glob("*.pkl")))
            return 0
        except Exception as e:
            logger.error(f"Error counting models: {e}")
            return 0
    
    def save_metrics(self, metrics: Dict) -> bool:
        """Save health metrics to database.
        
        Args:
            metrics: Dictionary of health metrics
            
        Returns:
            True if successful
        """
        try:
            query = """
            INSERT INTO system_health 
            (cpu_percent, memory_percent, disk_percent, prediction_latency_ms, 
             data_freshness_minutes, active_models)
            VALUES (:cpu_percent, :memory_percent, :disk_percent, :prediction_latency_ms, 
                    :data_freshness_minutes, :active_models)
            """
            
            with db_ops.db.get_session() as session:
                from sqlalchemy import text
                session.execute(text(query), {
                    'cpu_percent': metrics.get('cpu_percent'),
                    'memory_percent': metrics.get('memory_percent'),
                    'disk_percent': metrics.get('disk_percent'),
                    'prediction_latency_ms': metrics.get('prediction_latency_ms'),
                    'data_freshness_minutes': metrics.get('data_freshness_minutes'),
                    'active_models': metrics.get('active_models')
                })
                session.commit()
            
            logger.debug("System health metrics saved")
            return True
            
        except Exception as e:
            logger.error(f"Error saving health metrics: {e}")
            return False
    
    def record_health_check(self, prediction_latency_ms: Optional[int] = None):
        """Collect and save current health metrics.
        
        Args:
            prediction_latency_ms: Time taken for predictions
        """
        metrics = self.collect_metrics(prediction_latency_ms)
        if metrics:
            self.save_metrics(metrics)
    
    def get_recent_health_summary(self, hours: int = 24) -> Dict:
        """Get summary of recent health metrics.
        
        Args:
            hours: Hours to look back
            
        Returns:
            Dictionary with health summary
        """
        try:
            query = """
            SELECT 
                AVG(cpu_percent) as avg_cpu,
                MAX(cpu_percent) as max_cpu,
                AVG(memory_percent) as avg_memory,
                MAX(memory_percent) as max_memory,
                AVG(disk_percent) as avg_disk,
                AVG(prediction_latency_ms) as avg_latency,
                AVG(data_freshness_minutes) as avg_freshness,
                COUNT(*) as checks_count
            FROM system_health
            WHERE timestamp >= NOW() - INTERVAL :hours
            """
            
            with db_ops.db.get_session() as session:
                from sqlalchemy import text
                result = session.execute(text(query), {'hours': f'{hours} hours'})
                row = result.fetchone()
            
            if row:
                return {
                    'avg_cpu': float(row[0]) if row[0] else 0,
                    'max_cpu': float(row[1]) if row[1] else 0,
                    'avg_memory': float(row[2]) if row[2] else 0,
                    'max_memory': float(row[3]) if row[3] else 0,
                    'avg_disk': float(row[4]) if row[4] else 0,
                    'avg_latency_ms': int(row[5]) if row[5] else 0,
                    'avg_data_freshness_minutes': int(row[6]) if row[6] else 0,
                    'health_checks': int(row[7]) if row[7] else 0
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting health summary: {e}")
            return {}


# Singleton instance
system_health_monitor = SystemHealthMonitor()