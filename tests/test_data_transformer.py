"""
Test data_transformer module.
"""

import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.data_transformer import (
    get_monthly_periods,
    backfill_missing_data
)
import pandas as pd


def test_get_monthly_periods():
    """Test monthly period generation."""
    start = datetime(2024, 1, 1)
    end = datetime(2024, 3, 31)
    
    periods = get_monthly_periods(start, end)
    
    assert len(periods) == 3, f"Expected 3 periods, got {len(periods)}"
    assert periods[0]['period_name'] == '2024-01'
    assert periods[1]['period_name'] == '2024-02'
    assert periods[2]['period_name'] == '2024-03'
    assert periods[0]['year'] == 2024
    assert periods[0]['month'] == 1
    
    print("✓ test_get_monthly_periods passed")


def test_backfill_missing_data():
    """Test backfilling logic."""
    # Create test data with gaps
    dates = pd.date_range('2024-01-01', '2024-01-10', freq='D')
    
    # Create data with missing day 5
    df = pd.DataFrame({
        'coin_id': ['bitcoin'] * 9,
        'timestamp': [d for i, d in enumerate(dates) if i != 4],  # Skip day 5
        'open': [100.0] * 9,
        'high': [110.0] * 9,
        'low': [90.0] * 9,
        'close': [105.0] * 9,
        'volume': [1000.0] * 9
    })
    
    # Backfill
    filled = backfill_missing_data(df, ffill_threshold_days=2, max_gap_days=7)
    
    assert not filled.empty, "Backfill returned empty DataFrame"
    assert len(filled) == 10, f"Expected 10 rows after backfill, got {len(filled)}"
    assert filled['coin_id'].unique()[0] == 'bitcoin'
    
    print("✓ test_backfill_missing_data passed")


def test_backfill_drops_large_gaps():
    """Test that backfill drops coins with large gaps."""
    dates = pd.date_range('2024-01-01', '2024-01-20', freq='D')
    
    # Create data with 10-day gap (should be dropped)
    df = pd.DataFrame({
        'coin_id': ['bitcoin'] * 10,
        'timestamp': list(dates[:5]) + list(dates[15:]),  # 10-day gap
        'open': [100.0] * 10,
        'high': [110.0] * 10,
        'low': [90.0] * 10,
        'close': [105.0] * 10,
        'volume': [1000.0] * 10
    })
    
    # Backfill with max_gap_days=7
    filled = backfill_missing_data(df, max_gap_days=7)
    
    assert filled.empty, "Expected empty DataFrame for large gap"
    
    print("✓ test_backfill_drops_large_gaps passed")


if __name__ == "__main__":
    print("Running data_transformer tests...")
    print()
    
    test_get_monthly_periods()
    test_backfill_missing_data()
    test_backfill_drops_large_gaps()
    
    print()
    print("All tests passed! ✓")
