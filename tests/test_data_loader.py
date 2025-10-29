"""
Tests for data_loader module.
Run with: pytest tests/test_data_loader.py
"""
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from data.data_loader import CryptoDataLoader
from data.config import DatabaseConfig, DataConfig

load_dotenv()

class TestDatabaseConfig:
    """Test DatabaseConfig class."""
    
    def test_init_with_defaults(self):
        """Test initialization with default values."""
        config = DatabaseConfig(os.getenv('DB_HOST', 'localhost'),
                                os.getenv('DB_PORT', 5432),
                                os.getenv('DB_NAME', 'crypto_trading')) 
        assert config.host == 'localhost'
        assert config.port == '5432'
        assert config.database == 'crypto_trading'
    
    def test_init_with_custom_values(self):
        """Test initialization with custom values."""
        config = DatabaseConfig(
            host='db.example.com',
            port=5433,
            database='test_db',
            user='test_user',
            password='test_pass'
        )
        assert config.host == 'db.example.com'
        assert config.port == 5433
        assert config.database == 'test_db'
        assert config.user == 'test_user'
    
    def test_connection_string(self):
        """Test connection string generation."""
        config = DatabaseConfig(
            host='localhost',
            port=5432,
            database='test_db',
            user='user',
            password='pass'
        )
        conn_str = config.get_connection_string()
        assert 'postgresql://user:pass@localhost:5432/test_db' == conn_str
    
    def test_to_dict_excludes_password(self):
        """Test to_dict method excludes password."""
        config = DatabaseConfig(password='secret')
        config_dict = config.to_dict()
        assert 'password' not in config_dict
        assert 'host' in config_dict


class TestDataConfig:
    """Test DataConfig class."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = DataConfig()
        assert config.window_size == 50
        assert config.feature_number == 3
        assert config.global_period == '1D'
    
    def test_feature_set_3(self):
        """Test feature set 3 (most common)."""
        config = DataConfig(feature_number=3)
        features = config.get_feature_list()
        assert features == ['close', 'high', 'low']
    
    def test_feature_set_5(self):
        """Test feature set 5 (full OHLCV)."""
        config = DataConfig(feature_number=5)
        features = config.get_feature_list()
        assert features == ['open', 'high', 'low', 'close', 'volume']
    
    def test_to_dict(self):
        """Test to_dict method."""
        config = DataConfig(window_size=30, feature_number=3)
        config_dict = config.to_dict()
        assert config_dict['window_size'] == 30
        assert config_dict['feature_number'] == 3
        assert 'features' in config_dict


class TestCryptoDataLoader:
    """Test CryptoDataLoader class."""
    
    @pytest.fixture
    def mock_engine(self):
        """Mock SQLAlchemy engine."""
        with patch('data.data_loader.create_engine') as mock_create_engine:
            mock_engine = MagicMock()
            mock_create_engine.return_value = mock_engine
            yield mock_engine
    
    @pytest.fixture
    def loader(self, mock_engine):
        """Create CryptoDataLoader instance with mocked engine."""
        return CryptoDataLoader('postgresql://test:test@localhost/test')
    
    def test_init(self, loader):
        """Test initialization."""
        assert loader.table_name == 'daily_market_data'
        assert loader.coin_table == 'index_monthly_constituents'
    
    def test_get_available_coins(self, loader, mock_engine):
        """Test get_available_coins method."""
        # Mock database response
        mock_result = [(f'coin{i}', 200) for i in range(5)]
        mock_conn = MagicMock()
        mock_conn.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        
        coins = loader.get_available_coins(
            start_date='2023-01-01',
            end_date='2023-12-31',
            min_data_points=100
        )
        
        assert len(coins) == 5
        assert coins[0] == 'coin0'
    
    def test_check_data_quality(self, loader):
        """Test data quality check."""
        # Create sample DataFrame
        df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100),
            'id': ['coin1'] * 50 + ['coin2'] * 50,
            'open': range(100),
            'high': range(100),
            'low': range(100),
            'close': range(100),
            'volume': range(100)
        })
        
        coin_ids = ['coin1', 'coin2']
        quality = loader.check_data_quality(df, coin_ids)
        
        assert quality['total_rows'] == 100
        assert quality['unique_coins'] == 2
        assert quality['expected_coins'] == 2
        assert isinstance(quality['date_range'], tuple)
        assert isinstance(quality['missing_values'], dict)
    
    def test_check_data_quality_with_missing_values(self, loader):
        """Test data quality check with missing values."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10),
            'id': ['coin1'] * 10,
            'open': [1, None, 3, 4, 5, None, 7, 8, 9, 10],
            'high': range(10),
            'low': range(10),
            'close': range(10),
            'volume': range(10)
        })
        
        quality = loader.check_data_quality(df, ['coin1'])
        
        assert quality['missing_values']['open'] == 2
        assert quality['missing_values']['high'] == 0


class TestDataQuality:
    """Test data quality functions."""
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        loader = CryptoDataLoader.__new__(CryptoDataLoader)
        loader.table_name = 'daily_market_data'
        
        df = pd.DataFrame()
        coin_ids = []
        quality = loader.check_data_quality(df, coin_ids)
        
        assert quality['total_rows'] == 0
        assert quality['unique_coins'] == 0
    
    def test_single_coin(self):
        """Test single coin data."""
        loader = CryptoDataLoader.__new__(CryptoDataLoader)
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10),
            'id': ['bitcoin'] * 10,
            'close': range(10),
            'high': range(10),
            'low': range(10)
        })
        
        quality = loader.check_data_quality(df, ['bitcoin'])
        assert quality['unique_coins'] == 1
        assert quality['expected_coins'] == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
