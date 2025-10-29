"""
Data transformer for cryptocurrency portfolio management.

This module orchestrates the data pipeline:
1. Loads monthly datasets with backfilling
2. Creates rolling windows for training
3. Splits data into train/validation/test sets
4. Caches processed data for fast reloading

Date ranges:
- Training: July 2018 - December 2022 (~54 months)
- Validation: January 2023 - December 2023 (12 months)
- Test: January 2024 - October 2025 (~22 months)
"""

import os
import json
import logging
import calendar
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd

from data.config import DatabaseConfig, DataConfig
from data.data_loader import load_ohlcv, load_index_constituents
from data.data_processor import create_tensor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def backfill_missing_data(
    df: pd.DataFrame,
    ffill_threshold_days: int = 2,
    max_gap_days: int = 7
) -> pd.DataFrame:
    """
    Fill missing data using hybrid forward fill + interpolation.
    
    Strategy:
    - Gaps ≤ 2 days: Forward fill (conservative)
    - Gaps > 2 days: Linear interpolation
    - Gaps > 7 days: Drop coin entirely
    
    Args:
        df: DataFrame with columns [coin_id, timestamp, open, high, low, close, volume]
        ffill_threshold_days: Max days for forward fill (default: 2)
        max_gap_days: Max gap before dropping coin (default: 7)
        
    Returns:
        DataFrame with backfilled data
    """
    filled_coins = []
    
    for coin in df['coin_id'].unique():
        coin_df = df[df['coin_id'] == coin].copy()
        coin_df = coin_df.sort_values('timestamp')
        
        # Create complete date range
        date_range = pd.date_range(
            coin_df['timestamp'].min(),
            coin_df['timestamp'].max(),
            freq='D'
        )
        
        # Check for gaps
        existing_dates = set(coin_df['timestamp'].dt.date)
        missing_dates = set(date_range.date) - existing_dates
        
        # Skip if gap too large
        if len(missing_dates) > max_gap_days:
            logger.debug(f"Dropping {coin}: {len(missing_dates)} missing days")
            continue
        
        # Reindex to full range
        coin_df = coin_df.set_index('timestamp')
        coin_df = coin_df.reindex(date_range)
        coin_df['coin_id'] = coin
        
        # Forward fill for small gaps
        coin_df = coin_df.ffill(limit=ffill_threshold_days)
        
        # Interpolate remaining gaps
        coin_df = coin_df.infer_objects(copy=False).interpolate(method='linear')
        
        # Check if still has NaN
        if coin_df.isna().any().any():
            logger.debug(f"Dropping {coin}: Still has NaN after backfilling")
            continue
        
        coin_df = coin_df.reset_index()
        coin_df = coin_df.rename(columns={'index': 'timestamp'})
        filled_coins.append(coin_df)
    
    if not filled_coins:
        return pd.DataFrame()
    
    return pd.concat(filled_coins, ignore_index=True)


def get_monthly_periods(
    start_date: datetime,
    end_date: datetime
) -> List[Dict[str, Any]]:
    """
    Get list of monthly periods between start and end dates.
    
    Args:
        start_date: Start date (e.g., datetime(2018, 7, 1))
        end_date: End date (e.g., datetime(2025, 10, 31))
        
    Returns:
        List of dicts with period info
    """
    periods = []
    current = datetime(start_date.year, start_date.month, 1)
    
    while current <= end_date:
        periods.append({
            'period_date': current,
            'period_name': current.strftime('%Y-%m'),
            'year': current.year,
            'month': current.month
        })
        # Move to next month
        if current.month == 12:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, current.month + 1, 1)
    
    return periods


def load_monthly_dataset(
    period_date: datetime,
    window_size: int = 50,
    db_config: Optional[DatabaseConfig] = None,
    data_config: Optional[DataConfig] = None
) -> Dict[str, Any]:
    """
    Load and process data for a single month.
    
    Args:
        period_date: First day of month
        window_size: Historical window size (default: 50)
        db_config: Database configuration
        data_config: Data configuration
        
    Returns:
        Dict with monthly dataset info
        
    Raises:
        ValueError: If fewer than 10 valid coins after backfilling
    """
    if db_config is None:
        db_config = DatabaseConfig()
    if data_config is None:
        data_config = DataConfig()
    
    # Calculate date range
    # Need window_size days before month start for first window
    start_date = period_date - timedelta(days=window_size)
    
    # Get last day of month
    last_day = calendar.monthrange(period_date.year, period_date.month)[1]
    end_date = datetime(period_date.year, period_date.month, last_day)
    
    logger.info(f"Loading {period_date.strftime('%Y-%m')}: "
                f"{start_date.date()} to {end_date.date()}")
    
    # Get top 15 coins as candidates (10 + 5 backup)
    constituents = load_index_constituents(period_date, db_config)
    candidates = constituents[:data_config.BACKUP_COINS]
    
    # Load OHLCV data
    df = load_ohlcv(db_config, candidates, start_date, end_date)
    
    if df.empty:
        raise ValueError(f"No data loaded for {period_date.strftime('%Y-%m')}")
    
    # Backfill missing data
    df = backfill_missing_data(
        df,
        ffill_threshold_days=data_config.BACKFILL_FFILL_THRESHOLD,
        max_gap_days=data_config.BACKFILL_MAX_GAP
    )
    
    if df.empty:
        raise ValueError(f"No valid data after backfilling for {period_date.strftime('%Y-%m')}")
    
    # Get valid coins and select top 10
    valid_coins = df['coin_id'].unique().tolist()
    
    if len(valid_coins) < data_config.MIN_COINS:
        raise ValueError(
            f"Only {len(valid_coins)} valid coins for {period_date.strftime('%Y-%m')}, "
            f"need {data_config.MIN_COINS}"
        )
    
    # Select top 10 by original index order
    selected_coins = [c for c in constituents if c in valid_coins][:data_config.MIN_COINS]
    
    # Filter to selected coins
    df = df[df['coin_id'].isin(selected_coins)].copy()
    df = df.sort_values(['coin_id', 'timestamp']).reset_index(drop=True)
    
    # Calculate number of days in month
    num_days = (end_date - period_date).days + 1
    
    return {
        'period_date': period_date,
        'period_name': period_date.strftime('%Y-%m'),
        'coins': selected_coins,
        'data': df,
        'num_days': num_days,
        'start_date': start_date,
        'end_date': end_date
    }


def create_rolling_windows(
    monthly_dataset: Dict[str, Any],
    window_size: int = 50,
    data_config: Optional[DataConfig] = None
) -> List[Dict[str, Any]]:
    """
    Create rolling windows from monthly dataset.
    
    Args:
        monthly_dataset: Output from load_monthly_dataset()
        window_size: Historical window size (default: 50)
        data_config: Data configuration
        
    Returns:
        List of window dicts with tensors
    """
    if data_config is None:
        data_config = DataConfig()
    
    df = monthly_dataset['data']
    coins = monthly_dataset['coins']
    period_date = monthly_dataset['period_date']
    num_days = monthly_dataset['num_days']
    
    features = data_config.get_feature_list()
    windows = []
    
    # Create one window per day in the month
    for day_offset in range(num_days):
        window_date = period_date + timedelta(days=day_offset)
        
        # Extract window: need window_size + 1 days
        # (window_size for state, +1 for reward)
        window_start = monthly_dataset['start_date'] + timedelta(days=day_offset)
        window_end = window_start + timedelta(days=window_size + 1)
        
        window_df = df[
            (df['timestamp'] >= window_start) &
            (df['timestamp'] < window_end)
        ].copy()
        
        # Skip if not enough data
        if len(window_df['timestamp'].unique()) < window_size + 1:
            logger.warning(
                f"Skipping window for {window_date.date()}: "
                f"insufficient data ({len(window_df['timestamp'].unique())} days)"
            )
            continue
        
        # Create tensor
        try:
            state_tensor, future_prices, coins_used = create_tensor(
                window_df,
                features,
                window_size
            )
            
            # Verify we got all 10 coins
            if len(coins_used) != len(coins):
                logger.warning(
                    f"Window {window_date.date()}: only {len(coins_used)}/10 coins, skipping"
                )
                continue
            
            windows.append({
                'date': window_date,
                'period_date': period_date,
                'period_name': monthly_dataset['period_name'],
                'state_tensor': state_tensor,
                'future_prices': future_prices,
                'coins': coins_used
            })
            
        except Exception as e:
            logger.warning(f"Failed to create window for {window_date.date()}: {e}")
            continue
    
    logger.info(
        f"Created {len(windows)} windows for {monthly_dataset['period_name']}"
    )
    
    return windows


def save_processed_windows(
    windows: List[Dict[str, Any]],
    split_name: str,
    cache_dir: str = "./cache"
) -> str:
    """
    Save processed windows to compressed NumPy format.
    
    Args:
        windows: List of window dicts
        split_name: 'train', 'val', or 'test'
        cache_dir: Cache directory
        
    Returns:
        Path to saved file
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    # Extract arrays
    state_tensors = np.stack([w['state_tensor'] for w in windows])
    future_prices = np.stack([w['future_prices'] for w in windows])
    
    # Metadata (can't save datetime in npz, so convert to strings)
    dates = [w['date'].isoformat() for w in windows]
    period_dates = [w['period_date'].isoformat() for w in windows]
    period_names = [w['period_name'] for w in windows]
    coins = [w['coins'] for w in windows]
    
    # Save
    filepath = os.path.join(cache_dir, f"{split_name}_windows.npz")
    np.savez_compressed(
        filepath,
        state_tensors=state_tensors,
        future_prices=future_prices,
        dates=dates,
        period_dates=period_dates,
        period_names=period_names,
        coins=coins
    )
    
    # Save metadata as JSON
    metadata = {
        'split_name': split_name,
        'num_windows': len(windows),
        'tensor_shape': state_tensors.shape,
        'date_range': (dates[0], dates[-1]),
        'created': datetime.now().isoformat()
    }
    
    metadata_path = os.path.join(cache_dir, f"{split_name}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    logger.info(f"Saved {len(windows)} windows to {filepath}")
    return filepath


def load_processed_windows(
    split_name: str,
    cache_dir: str = "./cache"
) -> Optional[List[Dict[str, Any]]]:
    """
    Load processed windows from cache.
    
    Args:
        split_name: 'train', 'val', or 'test'
        cache_dir: Cache directory
        
    Returns:
        List of window dicts or None if not cached
    """
    filepath = os.path.join(cache_dir, f"{split_name}_windows.npz")
    
    if not os.path.exists(filepath):
        return None
    
    try:
        # Load arrays
        data = np.load(filepath, allow_pickle=True)
        
        state_tensors = data['state_tensors']
        future_prices = data['future_prices']
        dates = data['dates']
        period_dates = data['period_dates']
        period_names = data['period_names']
        coins = data['coins']
        
        # Reconstruct window dicts
        windows = []
        for i in range(len(state_tensors)):
            windows.append({
                'date': datetime.fromisoformat(dates[i]),
                'period_date': datetime.fromisoformat(period_dates[i]),
                'period_name': str(period_names[i]),
                'state_tensor': state_tensors[i],
                'future_prices': future_prices[i],
                'coins': list(coins[i])
            })
        
        logger.info(f"Loaded {len(windows)} windows from {filepath}")
        return windows
        
    except Exception as e:
        logger.error(f"Failed to load cache from {filepath}: {e}")
        return None


def create_train_val_test_split(
    start_date: datetime = datetime(2018, 7, 1),
    end_date: datetime = datetime(2025, 10, 31),
    window_size: int = 50,
    db_config: Optional[DatabaseConfig] = None,
    data_config: Optional[DataConfig] = None,
    use_cache: bool = True,
    force_reload: bool = False
) -> Dict[str, Any]:
    """
    Create train/validation/test split with caching.
    
    Split strategy:
    - Train: July 2018 - December 2022 (~54 months)
    - Validation: January 2023 - December 2023 (12 months)
    - Test: January 2024 - October 2025 (~22 months)
    
    Args:
        start_date: Start date (default: Jul 1, 2018)
        end_date: End date (default: Oct 31, 2025)
        window_size: Historical window size (default: 50)
        db_config: Database configuration
        data_config: Data configuration
        use_cache: Use cached data if available (default: True)
        force_reload: Force reload from database (default: False)
        
    Returns:
        Dict with train/val/test windows and statistics
    """
    if data_config is None:
        data_config = DataConfig()
    
    cache_dir = data_config.CACHE_DIR
    
    # Try to load from cache
    if use_cache and not force_reload:
        logger.info("Attempting to load from cache...")
        train_windows = load_processed_windows('train', cache_dir)
        val_windows = load_processed_windows('val', cache_dir)
        test_windows = load_processed_windows('test', cache_dir)
        
        if train_windows and val_windows and test_windows:
            logger.info("Successfully loaded all splits from cache!")
            return {
                'train_windows': train_windows,
                'val_windows': val_windows,
                'test_windows': test_windows,
                'statistics': get_dataset_statistics(
                    train_windows, val_windows, test_windows
                )
            }
    
    # Process from scratch
    logger.info("Processing data from database...")
    
    # Get all monthly periods
    periods = get_monthly_periods(start_date, end_date)
    logger.info(f"Found {len(periods)} monthly periods")
    
    # Split periods by year
    train_periods = [p for p in periods if p['year'] <= data_config.TRAIN_END_YEAR]
    val_periods = [p for p in periods if p['year'] == data_config.VAL_END_YEAR]
    test_periods = [p for p in periods if p['year'] > data_config.VAL_END_YEAR]
    
    logger.info(f"Train: {len(train_periods)} months, "
                f"Val: {len(val_periods)} months, "
                f"Test: {len(test_periods)} months")
    
    # Process each split
    train_windows = []
    val_windows = []
    test_windows = []
    
    # Training
    logger.info("Processing training data...")
    for period in train_periods:
        try:
            monthly_data = load_monthly_dataset(
                period['period_date'], window_size, db_config, data_config
            )
            windows = create_rolling_windows(monthly_data, window_size, data_config)
            train_windows.extend(windows)
        except Exception as e:
            logger.warning(f"Skipping {period['period_name']}: {e}")
    
    # Validation
    logger.info("Processing validation data...")
    for period in val_periods:
        try:
            monthly_data = load_monthly_dataset(
                period['period_date'], window_size, db_config, data_config
            )
            windows = create_rolling_windows(monthly_data, window_size, data_config)
            val_windows.extend(windows)
        except Exception as e:
            logger.warning(f"Skipping {period['period_name']}: {e}")
    
    # Test
    logger.info("Processing test data...")
    for period in test_periods:
        try:
            monthly_data = load_monthly_dataset(
                period['period_date'], window_size, db_config, data_config
            )
            windows = create_rolling_windows(monthly_data, window_size, data_config)
            test_windows.extend(windows)
        except Exception as e:
            logger.warning(f"Skipping {period['period_name']}: {e}")
    
    # Validate we have data
    if not train_windows:
        raise ValueError("No training windows created!")
    if not val_windows:
        raise ValueError("No validation windows created!")
    if not test_windows:
        raise ValueError("No test windows created!")
    
    logger.info(f"Created {len(train_windows)} train, "
                f"{len(val_windows)} val, "
                f"{len(test_windows)} test windows")
    
    # Save to cache
    if use_cache:
        logger.info("Saving to cache...")
        save_processed_windows(train_windows, 'train', cache_dir)
        save_processed_windows(val_windows, 'val', cache_dir)
        save_processed_windows(test_windows, 'test', cache_dir)
    
    # Get statistics
    stats = get_dataset_statistics(train_windows, val_windows, test_windows)
    
    return {
        'train_windows': train_windows,
        'val_windows': val_windows,
        'test_windows': test_windows,
        'statistics': stats
    }


def get_dataset_statistics(
    train_windows: List[Dict[str, Any]],
    val_windows: List[Dict[str, Any]],
    test_windows: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Calculate dataset statistics.
    
    Args:
        train_windows: Training windows
        val_windows: Validation windows
        test_windows: Test windows
        
    Returns:
        Dict with statistics
    """
    def get_date_range(windows):
        if not windows:
            return None, None
        dates = [w['date'] for w in windows]
        return min(dates), max(dates)
    
    def get_unique_coins(windows):
        coins = set()
        for w in windows:
            coins.update(w['coins'])
        return coins
    
    train_start, train_end = get_date_range(train_windows)
    val_start, val_end = get_date_range(val_windows)
    test_start, test_end = get_date_range(test_windows)
    
    train_coins = get_unique_coins(train_windows)
    val_coins = get_unique_coins(val_windows)
    test_coins = get_unique_coins(test_windows)
    all_coins = train_coins | val_coins | test_coins
    
    tensor_shape = train_windows[0]['state_tensor'].shape if train_windows else None
    
    stats = {
        'num_train_windows': len(train_windows),
        'num_val_windows': len(val_windows),
        'num_test_windows': len(test_windows),
        'total_windows': len(train_windows) + len(val_windows) + len(test_windows),
        'train_date_range': (train_start, train_end),
        'val_date_range': (val_start, val_end),
        'test_date_range': (test_start, test_end),
        'train_unique_coins': len(train_coins),
        'val_unique_coins': len(val_coins),
        'test_unique_coins': len(test_coins),
        'all_unique_coins': len(all_coins),
        'tensor_shape': tensor_shape
    }
    
    return stats


if __name__ == "__main__":
    # Example usage
    logger.info("Creating train/val/test split...")
    
    try:
        split_data = create_train_val_test_split(
            use_cache=True,
            force_reload=False
        )
        
        stats = split_data['statistics']
        print("\n" + "="*60)
        print("DATASET STATISTICS")
        print("="*60)
        print(f"Train windows: {stats['num_train_windows']}")
        print(f"  Date range: {stats['train_date_range'][0].date()} to "
              f"{stats['train_date_range'][1].date()}")
        print(f"  Unique coins: {stats['train_unique_coins']}")
        print()
        print(f"Validation windows: {stats['num_val_windows']}")
        print(f"  Date range: {stats['val_date_range'][0].date()} to "
              f"{stats['val_date_range'][1].date()}")
        print(f"  Unique coins: {stats['val_unique_coins']}")
        print()
        print(f"Test windows: {stats['num_test_windows']}")
        print(f"  Date range: {stats['test_date_range'][0].date()} to "
              f"{stats['test_date_range'][1].date()}")
        print(f"  Unique coins: {stats['test_unique_coins']}")
        print()
        print(f"Total windows: {stats['total_windows']}")
        print(f"All unique coins: {stats['all_unique_coins']}")
        print(f"Tensor shape: {stats['tensor_shape']}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Failed to create split: {e}", exc_info=True)
