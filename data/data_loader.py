"""
Simple data loader for cryptocurrency OHLCV data from PostgreSQL.
"""
import os
import pandas as pd
from sqlalchemy import create_engine, text
from typing import List, Optional, Dict

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


def load_ohlcv(
    coins: List[str],
    start_date: str,
    end_date: str,
    db_config: Optional[DatabaseConfig] = None
) -> pd.DataFrame:
    """
    Load OHLCV data from database.
    
    Args:
        coins: List of coin IDs (e.g., ['bitcoin', 'ethereum'])
        start_date: Start date (YYYY-MM-DD format)
        end_date: End date (YYYY-MM-DD format)
        db_config: Database configuration (if None, reads from env)
        
    Returns:
        DataFrame with columns: [coin_id, timestamp, open, high, low, close, volume]
    """
    if db_config is None:
        db_config = DatabaseConfig()
    
    engine = create_engine(db_config.get_connection_string())
    
    query = text("""
        SELECT 
            id as coin_id,
            timestamp,
            open,
            high,
            low,
            price as close,
            volume
        FROM daily_market_data
        WHERE id = ANY(:coins)
          AND timestamp >= :start_date
          AND timestamp <= :end_date
        ORDER BY id, timestamp
    """)
    
    with engine.connect() as conn:
        df = pd.read_sql(
            query,
            conn,
            params={
                'coins': coins,
                'start_date': start_date,
                'end_date': end_date
            }
        )
    
    return df


def get_available_coins(db_config: Optional[DatabaseConfig] = None) -> List[str]:
    """
    Get list of all available coin IDs in database.
    
    Args:
        db_config: Database configuration (if None, reads from env)
        
    Returns:
        List of coin IDs
    """
    if db_config is None:
        db_config = DatabaseConfig()
    
    engine = create_engine(db_config.get_connection_string())
    
    query = text("SELECT DISTINCT id FROM daily_market_data ORDER BY id")
    
    with engine.connect() as conn:
        result = conn.execute(query)
        coins = [row[0] for row in result]
    
    return coins


def load_index_constituents(
    period_date: str,
    db_config: Optional[DatabaseConfig] = None
) -> List[str]:
    """
    Load coins that were in the index for a given period.
    
    Args:
        period_date: Period start date (YYYY-MM-DD format)
        db_config: Database configuration (if None, reads from env)
        
    Returns:
        List of coin IDs in the index for that period
    """
    if db_config is None:
        db_config = DatabaseConfig()
    
    engine = create_engine(db_config.get_connection_string())
    
    query = text("""
        SELECT coin_id
        FROM index_monthly_constituents
        WHERE period_start_date = :period_date
        ORDER BY initial_weight_at_rebalance DESC
    """)
    
    with engine.connect() as conn:
        result = conn.execute(query, {'period_date': period_date})
        coins = [row[0] for row in result]
    
    return coins


if __name__ == "__main__":
    # Example usage
    from dotenv import load_dotenv
    load_dotenv()
    
    # Get available coins
    coins = get_available_coins()
    print(f"Available coins: {len(coins)}")
    print(f"First 10: {coins[:10]}")
    
    # Load data for a few coins
    if len(coins) >= 2:
        df = load_ohlcv(
            coins=coins[:2],
            start_date='2024-01-01',
            end_date='2024-01-31'
        )
        print(f"\nLoaded {len(df)} rows")
        print(df.head())
