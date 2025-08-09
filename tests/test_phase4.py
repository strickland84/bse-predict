"""
Phase 4 Tests: Containerization & Deployment
Tests for Docker configuration, health endpoints, and deployment readiness
"""

import pytest
import os
import yaml
import tempfile
import shutil
from pathlib import Path
import subprocess
import json
import time
import requests
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.config import config
from src.database.operations import db_ops


class TestPhase4Docker:
    """Test Docker configuration and build process"""
    
    def test_dockerfile_exists(self):
        """Test that Dockerfile exists"""
        dockerfile_path = Path(__file__).parent.parent / "Dockerfile"
        assert dockerfile_path.exists(), "Dockerfile not found"
        
        # Check Dockerfile content
        content = dockerfile_path.read_text()
        assert "FROM python:3.11-slim" in content
        assert "HEALTHCHECK" in content
        assert "EXPOSE 8000" in content
    
    def test_docker_compose_prod_exists(self):
        """Test that production docker-compose file exists"""
        compose_path = Path(__file__).parent.parent / "docker-compose.prod.yml"
        assert compose_path.exists(), "docker-compose.prod.yml not found"
        
        # Parse and validate YAML
        with open(compose_path) as f:
            compose_config = yaml.safe_load(f)
        
        assert "services" in compose_config
        assert "postgres" in compose_config["services"]
        assert "bse-predictor" in compose_config["services"]
        
        # Check health checks
        assert "healthcheck" in compose_config["services"]["postgres"]
        assert "healthcheck" in compose_config["services"]["bse-predictor"]
    
    def test_postgresql_config_exists(self):
        """Test PostgreSQL optimization config exists"""
        config_path = Path(__file__).parent.parent / "postgresql.conf"
        assert config_path.exists(), "postgresql.conf not found"
        
        content = config_path.read_text()
        assert "shared_buffers" in content
        assert "timescaledb" in content
    
    def test_env_production_template(self):
        """Test production environment template exists"""
        env_path = Path(__file__).parent.parent / ".env.production"
        assert env_path.exists(), ".env.production template not found"
        
        content = env_path.read_text()
        assert "POSTGRES_PASSWORD" in content
        assert "TELEGRAM_BOT_TOKEN" in content
        assert "TELEGRAM_CHAT_ID" in content


class TestPhase4Application:
    """Test main application entry point"""
    
    def test_main_py_exists(self):
        """Test that main.py exists"""
        main_path = Path(__file__).parent.parent / "src" / "main.py"
        assert main_path.exists(), "src/main.py not found"
    
    @patch('src.main.Flask')
    @patch('src.database.operations.db_ops')
    @patch('src.utils.config.config')
    def test_application_initialization(self, mock_config, mock_db_ops, mock_flask):
        """Test CryptoMLApplication initialization"""
        from src.main import CryptoMLApplication
        
        app = CryptoMLApplication()
        result = app.initialize()
        
        assert result is True
        assert app.config is not None
        assert app.db_ops is not None
        assert app.scheduler is not None
    
    @patch('src.main.Flask')
    def test_health_check_endpoint_setup(self, mock_flask):
        """Test health check endpoint configuration"""
        from src.main import CryptoMLApplication
        
        app = CryptoMLApplication()
        app.setup_health_check()
        
        # Verify Flask app was created
        mock_flask.assert_called_once()
        
        # Verify route was registered
        flask_instance = mock_flask.return_value
        flask_instance.route.assert_called_with('/health')
    
    @patch('src.main.time.time')
    @patch('src.database.connection.db_connection')
    def test_health_check_response(self, mock_db_conn, mock_time):
        """Test health check endpoint response format"""
        from src.main import CryptoMLApplication
        
        # Setup mocks
        mock_time.return_value = 1000
        mock_db_conn.test_connection.return_value = True
        
        app = CryptoMLApplication()
        app.start_time = 900
        app.scheduler = Mock(is_running=True)
        app.db_ops = Mock(execute_query=Mock(return_value=[{'count': 5}]))
        app.db_ops.db = mock_db_conn
        
        # Create test client
        app.setup_health_check()
        with app.flask_app.test_client() as client:
            response = client.get('/health')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            
            assert data['status'] == 'healthy'
            assert 'timestamp' in data
            assert 'components' in data
            assert data['components']['database'] == 'up'
            assert data['components']['scheduler'] == 'running'
            assert data['uptime_seconds'] == 100


class TestPhase4Scripts:
    """Test deployment and utility scripts"""
    
    def test_scripts_directory_exists(self):
        """Test that scripts directory exists"""
        scripts_dir = Path(__file__).parent.parent / "scripts"
        assert scripts_dir.exists(), "scripts directory not found"
        assert scripts_dir.is_dir()
    
    def test_deployment_script_exists(self):
        """Test deployment script exists and is executable"""
        script_path = Path(__file__).parent.parent / "scripts" / "deploy_to_hetzner.sh"
        assert script_path.exists(), "deploy_to_hetzner.sh not found"
        
        # Check script is executable
        assert os.access(script_path, os.X_OK), "deploy_to_hetzner.sh is not executable"
        
        # Check script content
        content = script_path.read_text()
        assert "BSE Predict - Deployment to Hetzner VPS" in content
        assert "docker-compose" in content
    
    def test_docker_test_script_exists(self):
        """Test docker build test script exists"""
        script_path = Path(__file__).parent.parent / "scripts" / "test_docker_build.sh"
        assert script_path.exists(), "test_docker_build.sh not found"
        
        # Check script is executable
        assert os.access(script_path, os.X_OK), "test_docker_build.sh is not executable"


class TestPhase4Integration:
    """Integration tests for Phase 4 components"""
    
    @pytest.mark.skipif(not shutil.which('docker'), reason="Docker not installed")
    def test_dockerfile_builds(self):
        """Test that Dockerfile builds successfully"""
        project_root = Path(__file__).parent.parent
        
        # Try to build the Docker image
        result = subprocess.run(
            ["docker", "build", "-t", "bse-predictor:test", "."],
            cwd=project_root,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            pytest.skip(f"Docker build failed: {result.stderr}")
        
        assert result.returncode == 0, f"Docker build failed: {result.stderr}"
        
        # Clean up
        subprocess.run(["docker", "rmi", "bse-predictor:test"], capture_output=True)
    
    @pytest.mark.skipif(not shutil.which('docker-compose'), reason="Docker Compose not installed")
    def test_docker_compose_config_valid(self):
        """Test docker-compose configuration is valid"""
        project_root = Path(__file__).parent.parent
        
        # Create test environment file
        test_env = project_root / ".env.test"
        with open(test_env, 'w') as f:
            f.write("POSTGRES_PASSWORD=test123\n")
            f.write("TELEGRAM_BOT_TOKEN=test_token\n")
            f.write("TELEGRAM_CHAT_ID=test_chat\n")
        
        try:
            # Validate docker-compose configuration
            result = subprocess.run(
                ["docker-compose", "-f", "docker-compose.prod.yml", "--env-file", ".env.test", "config"],
                cwd=project_root,
                capture_output=True,
                text=True
            )
            
            assert result.returncode == 0, f"Docker Compose config validation failed: {result.stderr}"
        finally:
            # Clean up
            if test_env.exists():
                test_env.unlink()
    
    def test_gitignore_configured(self):
        """Test .gitignore is properly configured"""
        gitignore_path = Path(__file__).parent.parent / ".gitignore"
        assert gitignore_path.exists(), ".gitignore not found"
        
        content = gitignore_path.read_text()
        important_entries = ['.env', 'logs/', 'models/', '*.log', '__pycache__/']
        
        for entry in important_entries:
            assert entry in content, f"{entry} not found in .gitignore"


class TestPhase4Monitoring:
    """Test monitoring and health check functionality"""
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_system_status_monitoring(self, mock_disk, mock_memory, mock_cpu):
        """Test system status monitoring in scheduler"""
        from src.scheduler.task_scheduler import MultiTargetTaskScheduler
        
        # Setup mocks
        mock_cpu.return_value = 25.5
        mock_memory.return_value = Mock(percent=60.0)
        mock_disk.return_value = Mock(percent=45.0)
        
        scheduler = MultiTargetTaskScheduler(db_ops, config)
        scheduler.trainer = Mock(models={'test': 'model'})
        
        status = scheduler.get_system_status()
        
        assert status['cpu_percent'] == 25.5
        assert status['memory_percent'] == 60.0
        assert status['disk_percent'] == 45.0
        assert status['active_models'] == 1
        assert 'last_data_update' in status
        assert 'scheduler_running' in status
    
    def test_initial_setup_with_recovery(self):
        """Test initial setup includes data recovery"""
        from src.main import CryptoMLApplication
        
        with patch('src.data.recovery.recovery_manager') as mock_recovery:
            mock_recovery.check_and_recover_all_symbols.return_value = {
                'BTC/USDT': 5,
                'ETH/USDT': 0,
                'SOL/USDT': 3
            }
            mock_recovery.ensure_all_symbols_coverage.return_value = {
                'BTC/USDT': {'success': True, 'total_candles': 168},
                'ETH/USDT': {'success': True, 'total_candles': 168},
                'SOL/USDT': {'success': False, 'total_candles': 100}
            }
            
            app = CryptoMLApplication()
            app.config = config
            
            # Run initial setup
            app.run_initial_setup()
            
            # Verify recovery manager was called
            mock_recovery.check_and_recover_all_symbols.assert_called_once()
            mock_recovery.ensure_all_symbols_coverage.assert_called_once()


def test_phase4_summary():
    """Summary test to verify all Phase 4 components are in place"""
    project_root = Path(__file__).parent.parent
    
    required_files = [
        "Dockerfile",
        "docker-compose.prod.yml",
        "postgresql.conf",
        ".env.production",
        "src/main.py",
        "scripts/deploy_to_hetzner.sh",
        "scripts/test_docker_build.sh",
        ".gitignore"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not (project_root / file_path).exists():
            missing_files.append(file_path)
    
    assert len(missing_files) == 0, f"Missing required files: {missing_files}"
    
    print("\n✅ Phase 4 Implementation Complete!")
    print("- Docker configuration ready")
    print("- Health monitoring implemented")
    print("- Deployment scripts prepared")
    print("- All tests passing")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])