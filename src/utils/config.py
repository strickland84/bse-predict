"""Configuration management for BSE Predict."""
import os
import yaml
from pathlib import Path
from typing import Any, Dict, List
import logging

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager that loads from YAML and environment variables."""
    
    def __init__(self, config_path: str = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to config file. If None, uses default config.yaml
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file and override with environment variables."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Config file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing config file: {e}")
            raise
            
        # Override with environment variables
        config = self._override_with_env_vars(config)
        return config
    
    def _override_with_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Override configuration with environment variables."""
        # Database
        if os.getenv('DATABASE_URL'):
            config['database']['url'] = os.getenv('DATABASE_URL')
            
        # Telegram
        if os.getenv('TELEGRAM_BOT_TOKEN'):
            config['telegram']['bot_token'] = os.getenv('TELEGRAM_BOT_TOKEN')
        if os.getenv('TELEGRAM_CHAT_ID'):
            config['telegram']['chat_id'] = os.getenv('TELEGRAM_CHAT_ID')
            
        # Trading
        if os.getenv('EXCHANGE'):
            config['trading']['exchange'] = os.getenv('EXCHANGE')
            
        # Environment
        if os.getenv('ENVIRONMENT'):
            config['app']['environment'] = os.getenv('ENVIRONMENT')
            
        return config
    
    def _validate_config(self) -> None:
        """Validate configuration values."""
        required_keys = [
            'trading.assets',
            'trading.target_percentages',
            'database.url',
            'telegram.bot_token',
            'telegram.chat_id'
        ]
        
        for key in required_keys:
            if not self.get_nested(key):
                logger.warning(f"Missing required configuration: {key}")
    
    def get_nested(self, key_path: str, default: Any = None) -> Any:
        """Get nested configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path like 'database.url'
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
                
        return value
    
    @property
    def assets(self) -> List[str]:
        """Get trading assets."""
        return self.config['trading']['assets']
    
    @property
    def target_percentages(self) -> List[float]:
        """Get target percentages for predictions."""
        return self.config['trading']['target_percentages']
    
    @property
    def database_url(self) -> str:
        """Get database URL."""
        return self.config['database']['url']
    
    @property
    def telegram_config(self) -> Dict[str, Any]:
        """Get Telegram configuration."""
        return self.config['telegram']
    
    @property
    def exchange_name(self) -> str:
        """Get exchange name."""
        return self.config['trading']['exchange']
    
    @property
    def log_level(self) -> str:
        """Get logging level."""
        return self.config['logging']['level']
    
    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access to configuration."""
        return self.get_nested(key)
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config(path={self.config_path})"


# Global configuration instance
config = Config()
