"""
Configuration utilities for database connection and data loading.
"""

import os
from typing import Dict


class DatabaseConfig:
    """Database configuration - reads from environment variables."""
    
    def __init__(self):
        """Initialize database configuration from environment variables."""
        self.host = os.getenv('DB_HOST', 'localhost')
        self.port = int(os.getenv('DB_PORT', '5432'))
        self.database = os.getenv('DB_NAME')
        self.user = os.getenv('DB_USER', 'postgres')
        self.password = os.getenv('DB_PASSWORD', '')
    
    def get_connection_string(self) -> str:
        """Get SQLAlchemy connection string."""
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )
    
    def to_dict(self) -> Dict[str, str]:
        """Convert config to dictionary (excluding password)."""
        return {
            'host': self.host,
            'port': self.port,
            'database': self.database,
            'user': self.user
        }


class DataConfig:
    """Data loading and processing configuration."""
    
    # Feature sets based on Jiang et al. 2017
    FEATURE_SET_1 = ['close']
    FEATURE_SET_2 = ['close', 'volume']
    FEATURE_SET_3 = ['close', 'high', 'low']
    FEATURE_SET_4 = ['open', 'high', 'low', 'close']
    FEATURE_SET_5 = ['open', 'high', 'low', 'close', 'volume']
    
    # Backfilling configuration
    # Strategy: Forward fill for gaps ≤ 2 days, interpolate for longer gaps
    BACKFILL_FFILL_THRESHOLD = 2  # Days: use forward fill for gaps ≤ 2 days
    BACKFILL_MAX_GAP = 7          # Days: drop coins with gaps > 7 days
    
    # Monthly dataset configuration
    MIN_COINS = 10                # Minimum required coins per month
    BACKUP_COINS = 15             # Load top N coins as candidates (10 + 5 backup)
    
    # Train/validation/test split configuration (calendar years)
    TRAIN_END_YEAR = 2022         # Last year for training (Jul 2018 - Dec 2022)
    VAL_END_YEAR = 2023           # Last year for validation (2023)
    # Test: 2024 - present
    
    # Caching configuration
    CACHE_DIR = "./cache"
    USE_CACHE = True              # Enable caching by default
    
    def __init__(
        self,
        window_size: int = 50,
        feature_number: int = 3
    ):
        """
        Initialize data configuration.
        
        Args:
            window_size: Number of historical periods (default: 50)
            feature_number: Feature set to use (1-5, default: 3)
        """
        self.window_size = window_size
        self.feature_number = feature_number
    
    def get_feature_list(self):
        """Get list of features based on feature_number."""
        feature_sets = {
            1: self.FEATURE_SET_1,
            2: self.FEATURE_SET_2,
            3: self.FEATURE_SET_3,
            4: self.FEATURE_SET_4,
            5: self.FEATURE_SET_5
        }
        return feature_sets.get(self.feature_number, self.FEATURE_SET_3)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    db_config = DatabaseConfig()
    print(f"Database config: {db_config.to_dict()}")
    print(f"Connection string: {db_config.get_connection_string()}")
    
    data_config = DataConfig()
    print(f"\nFeatures: {data_config.get_feature_list()}")
