"""
Data processor for cryptocurrency RL portfolio management.
Transforms raw OHLCV data into normalized 3D tensors.
"""

import numpy as np
import pandas as pd
from typing import List


def create_tensor(
    df: pd.DataFrame,
    features: List[str],
    window_size: int
) -> tuple:
    """
    Create 3D tensor from OHLCV DataFrame with future prices for reward calculation.
    
    Args:
        df: DataFrame with columns [coin_id, timestamp, open, high, low, close, volume]
            Must contain window_size + 1 periods (state + future for reward)
        features: List of features to include (e.g., ['close', 'high', 'low'])
        window_size: Number of historical periods to include in state
        
    Returns:
        Tuple of (state_tensor, future_prices, coins_used):
            - state_tensor: [feature_number, coin_number, window_size] normalized by last close
            - future_prices: [coin_number] close prices at window_size + 1 for reward calculation
            - coins_used: List of coin IDs included in the tensor
    """
    # Sort by coin and timestamp
    df = df.sort_values(['coin_id', 'timestamp'])
    
    # Get unique coins
    coins = df['coin_id'].unique()
    coin_number = len(coins)
    
    # Get unique timestamps
    timestamps = df['timestamp'].unique()
    
    # Check if we have enough data (need window_size + 1 for state + future)
    if len(timestamps) < window_size + 1:
        raise ValueError(
            f"Not enough data: {len(timestamps)} timestamps < {window_size + 1} required"
        )
    
    # Check if we have enough data (need window_size + 1 for state + future)
    if len(timestamps) < window_size + 1:
        raise ValueError(
            f"Not enough data: {len(timestamps)} timestamps < {window_size + 1} required"
        )
    
    # Use the last window_size + 1 periods
    timestamps_with_future = timestamps[-(window_size + 1):]
    state_timestamps = timestamps_with_future[:-1]  # First window_size periods for state
    future_timestamp = timestamps_with_future[-1]   # Last period for reward
    
    # Filter dataframe
    df_all = df[df['timestamp'].isin(timestamps_with_future)]
    
    # Check which coins have complete data for all window_size + 1 periods
    coin_counts = df_all.groupby('coin_id')['timestamp'].nunique()
    valid_coins = coin_counts[coin_counts == window_size + 1].index.tolist()
    
    if len(valid_coins) == 0:
        raise ValueError("No coins have complete data for window_size + 1 periods")
    
    # Filter to only valid coins
    df_all = df_all[df_all['coin_id'].isin(valid_coins)]
    coins = valid_coins
    coin_number = len(coins)
    
    # Split into state data and future data
    df_state = df_all[df_all['timestamp'].isin(state_timestamps)]
    df_future = df_all[df_all['timestamp'] == future_timestamp]
    
    # Create 3D tensor for state: [features, coins, time]
    state_tensor = np.zeros((len(features), coin_number, window_size))
    
    # Create 3D tensor for state: [features, coins, time]
    state_tensor = np.zeros((len(features), coin_number, window_size))
    
    for f_idx, feature in enumerate(features):
        # Pivot to get [coins × time] matrix
        pivot = df_state.pivot(index='timestamp', columns='coin_id', values=feature)
        
        # Reindex to ensure we have all coins in order
        pivot = pivot.reindex(columns=coins)
        
        # Fill any missing values with forward fill, then backward fill
        pivot = pivot.ffill().bfill()
        
        # Transpose to [coins × time] and store
        state_tensor[f_idx] = pivot.T.values
    
    # Get future close prices for reward calculation
    future_close_pivot = df_future.pivot(index='timestamp', columns='coin_id', values='close')
    future_close_pivot = future_close_pivot.reindex(columns=coins)
    future_prices = future_close_pivot.values[0]  # [coin_number]
    
    # Normalize state tensor by last close price in the state window
    # Find close feature index
    if 'close' not in features:
        raise ValueError("'close' must be in features for normalization")
    
    close_idx = features.index('close')
    last_close = state_tensor[close_idx, :, -1]  # [coin_number]
    
    # Divide all features by last close (broadcast)
    state_tensor = state_tensor / last_close[None, :, None]
    
    return state_tensor, future_prices, coins


def validate_tensor(tensor: np.ndarray, expected_shape: tuple) -> bool:
    """
    Validate tensor shape and check for NaNs.
    
    Args:
        tensor: Tensor to validate
        expected_shape: Expected shape (feature_number, coin_number, window_size)
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If validation fails
    """
    if tensor.shape != expected_shape:
        raise ValueError(f"Shape mismatch: {tensor.shape} != {expected_shape}")
    
    if np.isnan(tensor).any():
        raise ValueError("Tensor contains NaN values")
    
    if np.isinf(tensor).any():
        raise ValueError("Tensor contains infinite values")
    
    return True


if __name__ == "__main__":
    # Example usage
    from dotenv import load_dotenv
    from data_loader import load_ohlcv, get_available_coins
    from config import DataConfig
    
    load_dotenv()
    
    # Get some coins
    coins = get_available_coins()
    print(f"Available coins: {len(coins)}")
    
    # Load data for first few coins
    test_coins = coins[:5]
    df = load_ohlcv(
        coins=test_coins,
        start_date='2024-01-01',
        end_date='2024-12-31'
    )
    
    print(f"\nLoaded {len(df)} rows for {len(test_coins)} coins")
    
    if len(df) > 0:
        # Create tensor with feature set 3 (close, high, low)
        config = DataConfig(window_size=50, feature_number=3)
        features = config.get_feature_list()
        
        state_tensor, future_prices, coins_used = create_tensor(df, features, config.window_size)
        print(f"Coins with complete data: {len(coins_used)}")
        print(f"Coins used: {coins_used}")
        
        print(f"\nState tensor shape: {state_tensor.shape}")
        print(f"Expected: [{len(features)}, {len(coins_used)}, {config.window_size}]")
        print(f"Future prices shape: {future_prices.shape}")
        print(f"Expected: [{len(coins_used)}]")
        
        # Validate
        validate_tensor(state_tensor, (len(features), len(coins_used), config.window_size))
        print("✓ State tensor validated successfully")
        
        # Show sample values
        print(f"\nSample values (first coin, last 3 time periods):")
        for i, feature in enumerate(features):
            print(f"  {feature}: {state_tensor[i, 0, -3:]}")
        
        print(f"\nFuture close prices (for reward calculation):")
        print(f"  First coin: {future_prices[0]:.2f}")
        
        # Calculate price relatives (what reward calculation will use)
        last_state_close = state_tensor[features.index('close'), :, -1]
        # Note: last_state_close is already normalized to 1.0
        # So we need to unnormalize or work with the original last close
        print(f"\nLast state close (normalized): {last_state_close[0]:.4f}")
    else:
        print("No data loaded for testing")
