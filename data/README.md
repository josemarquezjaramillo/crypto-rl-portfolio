# Data Module - Crypto RL Portfolio Management

Simplified data loading and processing pipeline for cryptocurrency portfolio management with Deep Reinforcement Learning.

## Overview

This module implements a clean, focused data pipeline following the framework established by **Jiang et al. (2017)** in their seminal paper ["A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem"](https://arxiv.org/abs/1706.10059). The implementation is designed for **daily rebalancing** using daily OHLCV data.

### Design Philosophy

Our implementation follows three key principles:

1. **Separation of Concerns**: Each module has a single, well-defined responsibility
2. **Simplicity First**: Start with daily data before considering higher-frequency trading
3. **Research-Driven**: Design decisions are grounded in the academic literature

## Architecture

```
data_loader.py → data_processor.py → data_transformer.py → RL Environment
    ↓                   ↓                     ↓
  Database          Tensors            Train/Test Splits
```

### Module Responsibilities

- **`data_loader.py`**: Query PostgreSQL database, return pandas DataFrames
- **`data_processor.py`**: Transform DataFrames into normalized 3D tensors
- **`data_transformer.py`**: Handle train/test splits, rolling windows, orchestration
- **`config.py`**: Configuration management (database, features, hyperparameters)

## Theoretical Foundation

### The Jiang et al. (2017) Framework

Jiang's framework uses a **3D tensor representation** of historical price data as input to a deep reinforcement learning agent:

**Tensor Shape**: `[feature_number, coin_number, window_size]`

- **Axis 0 (Features)**: OHLCV data (Open, High, Low, Close, Volume)
- **Axis 1 (Assets)**: Different cryptocurrencies in the portfolio
- **Axis 2 (Time)**: Historical time periods (lookback window)

**Key Insight**: The framework uses **raw price data** normalized by the last close price, not traditional technical indicators. The CNN/RNN architecture learns relevant patterns directly from the data.

### Our Implementation Decisions

#### 1. Time Granularity: Daily vs. Intraday

**Jiang et al. used**: 30-minute periods (1800 seconds)
- `window_size = 50` → 25 hours of history
- Rebalancing every 30 minutes

**We use**: Daily periods (86400 seconds)
- `window_size = 50` → 50 days of history  
- Rebalancing once per day

**Rationale**:
- ✅ **Practical for retail investors**: Daily rebalancing is realistic; 30-min rebalancing incurs excessive transaction costs
- ✅ **Less noise**: Daily data smooths out intraday volatility
- ✅ **Available data**: We have complete daily OHLCV data in our database
- ✅ **Faster iteration**: 48x less data means faster training and experimentation
- ✅ **Alignment with index rebalancing**: Our monthly index constituents naturally align with daily periods

**Citation**: Jiang, Z., Xu, D., & Liang, J. (2017). Section 3.1: "The global period...should be a multiple of 300 seconds"

#### 2. Coin Selection: Index Constituents vs. Volume Ranking

**Jiang et al. used**: Top N coins by trading volume over the entire backtest period
- Fixed universe of coins
- Selected once at the beginning

**We use**: Monthly index constituents with top 10 by market cap
- Dynamic universe that changes monthly
- Pre-selected based on historical market cap (prevents lookahead bias)

**Rationale**:
- ✅ **Prevents survivorship bias**: Coins enter/exit the index realistically
- ✅ **Prevents lookahead bias**: Constituents selected using only past data
- ✅ **Mimics real index funds**: Monthly rebalancing is industry standard (e.g., S&P rebalances quarterly)
- ✅ **Better risk management**: Market cap weighting is more stable than volume

**Database Schema**:
```sql
CREATE TABLE index_monthly_constituents (
    period_start_date date,
    coin_id varchar,
    initial_market_cap_at_rebalance numeric,
    initial_weight_at_rebalance numeric
);
```

#### 3. Reward Calculation: Next-Period Returns

**The RL Environment Loop**:

```
Time t:
  State: 50-day OHLCV tensor (normalized)
  ↓
  Agent decides: Portfolio weights [w_1, w_2, ..., w_10]
  ↓
  Hold for 1 day
  ↓
Time t+1:
  Reward: Portfolio return = Σ(w_i × price_relative_i)
  where price_relative_i = close[t+1] / close[t]
  ↓
  State: New 50-day window (shifted by 1 day)
```

**Implementation**:
- Load `window_size + 1` periods from database
- Use first `window_size` periods for state tensor
- Use period `window_size + 1` for reward calculation

**Citation**: Jiang et al. (2017), Section 3.2: "The reward function is the logarithmic return of the portfolio"

#### 4. Normalization: Relative Prices

**Method**: Divide all prices by the last close price in the window

```python
normalized_prices = raw_prices / last_close_price
```

**Result**: Last close price = 1.0, all other prices are relative to it

**Rationale**:
- ✅ **Scale-invariant**: Works for both $0.01 and $10,000 coins
- ✅ **Focuses on returns**: Agent learns percentage changes, not absolute prices
- ✅ **Numerical stability**: Values centered around 1.0 help neural network training

**Citation**: Jiang et al. (2017), Section 3.3: "Prices are normalized by dividing by the last close price"

## Module Documentation

### `data_loader.py`

**Purpose**: Simple database queries, return raw data

**Functions**:
```python
load_ohlcv(coins, start_date, end_date) -> pd.DataFrame
    # Load OHLCV data for specified coins and date range
    
get_available_coins() -> List[str]
    # Get all coin IDs in database
    
load_index_constituents(period_date) -> List[str]
    # Get coins in index for a given month
```

**Design Decision**: Keep it simple - just SQL queries, no preprocessing
- Single responsibility: database access
- Easy to test and debug
- No hidden transformations

### `data_processor.py`

**Purpose**: Transform DataFrames into normalized tensors

**Functions**:
```python
create_tensor(df, features, window_size) -> (state_tensor, future_prices, coins_used)
    # Input: DataFrame with window_size + 1 periods
    # Output: 
    #   - state_tensor: [features × coins × window_size]
    #   - future_prices: [coins] for reward calculation
    #   - coins_used: list of coin IDs
    
validate_tensor(tensor, expected_shape) -> bool
    # Check for NaN, Inf, shape mismatches
```

**Key Operations**:
1. **Pivot**: Long DataFrame → wide format (coins as columns)
2. **Filter**: Only include coins with complete data
3. **Split**: Separate state (first 50 days) from future (day 51)
4. **Normalize**: Divide by last close price in state window
5. **Validate**: Ensure no NaN/Inf values

**Design Decision**: Pure transformation - doesn't know about database or RL environment

### `config.py`

**Purpose**: Centralized configuration management

**Classes**:
```python
DatabaseConfig()
    # Reads from environment variables
    # Returns connection string
    
DataConfig(window_size=50, feature_number=3)
    # Defines feature sets (Jiang's 5 variants)
    # FEATURE_SET_3 = ['close', 'high', 'low']  # Default
```

**Feature Sets** (from Jiang et al. 2017):
- **Set 1**: `['close']` - Minimal
- **Set 2**: `['close', 'volume']` - Not well supported in original paper
- **Set 3**: `['close', 'high', 'low']` - **Recommended starting point**
- **Set 4**: `['open', 'high', 'low', 'close']` - Full OHLC
- **Set 5**: `['open', 'high', 'low', 'close', 'volume']` - Full OHLCV

**Design Decision**: Start with Set 3 - captures price range and volatility without volume noise

### `data_transformer.py`

**Purpose**: Orchestration, train/val/test splits, rolling windows, caching

**Functions**:
```python
backfill_missing_data(df, date_range, coins) -> pd.DataFrame
    # Hybrid backfilling strategy:
    # - Forward fill ≤2 days (weekends, minor gaps)
    # - Linear interpolate 3-7 days (medium gaps)
    # - Drop coins with gaps >7 days (unreliable data)
    
get_monthly_periods(start_year, start_month, end_year, end_month) -> List[tuple]
    # Generate list of monthly periods: [(year, month), ...]
    # Example: [(2018, 7), (2018, 8), ..., (2025, 10)]
    
load_monthly_dataset(period_year, period_month, window_size) -> Dict
    # Load data for one month with extended window
    # Returns: {
    #   'states': [N_days, features, coins, window_size],
    #   'future_prices': [N_days, coins],
    #   'coins': [coins],
    #   'dates': [N_days]
    # }
    
create_rolling_windows(period_year, period_month, window_size) -> Dict
    # Create daily rolling windows for one month
    # Loads window_size days before month start
    # Returns one window per day in the month
    
save_processed_windows(windows, split_name, cache_dir) -> None
    # Save tensors to NPZ compressed format
    # Optimized for Google Drive sharing
    
load_processed_windows(split_name, cache_dir) -> Dict
    # Load cached tensors (~100x faster than database)
    
create_train_val_test_split(use_cache=True) -> Dict
    # Main orchestration function
    # Returns: {
    #   'train': Dict with states, future_prices, coins, dates
    #   'val': Dict with states, future_prices, coins, dates  
    #   'test': Dict with states, future_prices, coins, dates
    # }
    
get_dataset_statistics(dataset) -> Dict
    # Calculate dataset statistics for validation
```

**Key Operations**:
1. **Monthly Periods**: Generate all months from Jul 2018 to Oct 2025
2. **Extended Windows**: Load window_size days before each month
3. **Backfilling**: Hybrid strategy preserves data quality
4. **Rolling Windows**: One window per day in month (~31 windows/month)
5. **Fixed Coins**: Always exactly 10 coins per month
6. **Caching**: Save/load processed tensors for fast iteration

**Design Decision**: Separates data preparation from RL environment - handles all preprocessing once

## Database Schema

### `daily_market_data`
```sql
CREATE TABLE daily_market_data (
    id text,                          -- Coin identifier (e.g., 'bitcoin')
    timestamp timestamp with time zone, -- Date of observation
    price numeric,                     -- Representative price
    market_cap numeric,                -- Market capitalization
    volume numeric,                    -- 24h trading volume
    open numeric,                      -- Opening price
    high numeric,                      -- Highest price
    low numeric,                       -- Lowest price
    close numeric,                     -- Closing price
    PRIMARY KEY (id, timestamp)
);
```

### `index_monthly_constituents`
```sql
CREATE TABLE index_monthly_constituents (
    period_start_date date,                    -- First day of month
    coin_id varchar,                           -- Coin identifier
    initial_market_cap_at_rebalance numeric,   -- Market cap at selection
    initial_weight_at_rebalance numeric,       -- Portfolio weight (0-1)
    PRIMARY KEY (period_start_date, coin_id)
);
```

**Design Rationale**: 
- `index_monthly_constituents` enables realistic backtesting with time-appropriate coin selection
- Prevents lookahead bias by using only historical market caps
- Sorted by `initial_weight_at_rebalance DESC` to easily select top N coins

## Usage Examples

### Basic Usage

```python
from dotenv import load_dotenv
from data.data_loader import load_ohlcv, load_index_constituents
from data.data_processor import create_tensor
from data.config import DataConfig

load_dotenv()

# 1. Get coins for a given month
coins = load_index_constituents('2024-01-01')[:10]  # Top 10

# 2. Load OHLCV data (need window_size + 1 periods)
df = load_ohlcv(
    coins=coins,
    start_date='2023-11-01',  # 50 + 1 = 51 days before
    end_date='2024-01-01'
)

# 3. Create tensor
config = DataConfig(window_size=50, feature_number=3)
state, future_prices, coins_used = create_tensor(
    df, 
    config.get_feature_list(), 
    config.window_size
)

# 4. Use in RL environment
# state: [3, 10, 50] - input to neural network
# future_prices: [10] - for reward calculation after agent acts
```

### Rolling Window for Backtesting

```python
from data.data_transformer import create_train_val_test_split, get_dataset_statistics

# Load or generate full dataset with train/val/test splits
dataset = create_train_val_test_split(use_cache=True)

# Dataset structure:
# {
#   'train': {
#       'states': [N_train, 3, 10, 50],      # State tensors
#       'future_prices': [N_train, 10],      # For reward calculation
#       'coins': [N_train, 10],               # Coin IDs per window
#       'dates': [N_train]                    # Timestamp per window
#   },
#   'val': {...},    # Same structure
#   'test': {...}    # Same structure
# }

# Get statistics
stats = get_dataset_statistics(dataset['train'])
print(f"Training windows: {stats['num_windows']}")
print(f"Date range: {stats['start_date']} to {stats['end_date']}")
print(f"Unique coins: {stats['unique_coins']}")

# Use in RL training loop
for epoch in range(num_epochs):
    for i in range(len(dataset['train']['states'])):
        state = dataset['train']['states'][i]          # [3, 10, 50]
        future_prices = dataset['train']['future_prices'][i]  # [10]
        
        # Agent selects action (portfolio weights)
        action = agent.select_action(state)
        
        # Calculate reward
        reward = calculate_portfolio_return(action, future_prices)
        
        # Update agent
        agent.update(state, action, reward)
```

### Complete Pipeline Example

```python
from dotenv import load_dotenv
from data.data_transformer import create_train_val_test_split
from data.config import DataConfig

load_dotenv()

# First run: Load from database and cache (~10-30 minutes)
print("Loading full dataset from database...")
dataset = create_train_val_test_split(use_cache=False)

# Subsequent runs: Load from cache (~5-10 seconds)
print("Loading from cache...")
dataset = create_train_val_test_split(use_cache=True)

# Access different splits
train_data = dataset['train']
val_data = dataset['val']
test_data = dataset['test']

# Expected sizes (approximate):
# Train: ~1,620 windows (Jul 2018 - Dec 2022: 54 months × ~30 days)
# Val: ~360 windows (2023: 12 months × ~30 days)
# Test: ~660 windows (2024 - Oct 2025: 22 months × ~30 days)

print(f"Train: {len(train_data['states'])} windows")
print(f"Val: {len(val_data['states'])} windows")
print(f"Test: {len(test_data['states'])} windows")
```

## Installation

1. **Install dependencies**:
```bash
pip install -r requirements-data.txt
```

2. **Configure database**:
```bash
cp .env.example .env
# Edit .env with your PostgreSQL credentials:
# DB_HOST=localhost
# DB_PORT=5432
# DB_NAME=crypto_data
# DB_USER=postgres
# DB_PASSWORD=your_password
```

3. **Verify connection**:
```bash
python scripts/verify_database.py
```

## Testing

Run the test suite:
```bash
python -m pytest tests/ -v
```

Run individual module tests:
```bash
# Test data loader
python data/data_loader.py

# Test data processor  
python data/data_processor.py

# Test data transformer
python -m pytest tests/test_data_transformer.py -v
```

Expected test results:
```
tests/test_data_transformer.py::test_get_monthly_periods PASSED
tests/test_data_transformer.py::test_backfill_missing_data PASSED
tests/test_data_transformer.py::test_backfill_drops_large_gaps PASSED
```

## Key Design Decisions Summary

| Decision | Our Choice | Jiang et al. | Rationale |
|----------|-----------|--------------|-----------|
| **Time Period** | Daily (86400s) | 30-min (1800s) | Practical for retail, less noise, faster iteration |
| **Window Size** | 50 days | 50 periods | Same relative lookback, different time scale |
| **Coin Selection** | Monthly index (top 10 by market cap) | Fixed by volume | Prevents bias, mimics real indices |
| **Coin Count** | Fixed 10 per month | Fixed set | Neural network requires consistent dimensions |
| **Rolling Windows** | Extended (load before month) | Continuous | Adapted to monthly boundaries |
| **Backfilling** | Hybrid (ffill/interpolate/drop) | Not specified | Pragmatic data quality approach |
| **Data Split** | Year-based (2018-2022/2023/2024-2025) | Percentage-based (50/25/25) | Better for thesis, seasonal patterns |
| **Caching** | NPZ compressed | Not implemented | Fast iteration, team sharing |
| **Rebalancing** | Daily | Every 30 min | Realistic transaction costs |
| **Features** | Set 3: [close, high, low] | Sets 1-5 available | Good balance of information vs. complexity |
| **Normalization** | Divide by last close | Same | Scale-invariant, focuses on returns |

## References

1. **Jiang, Z., Xu, D., & Liang, J. (2017)**. "A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem." arXiv:1706.10059
   - **Sections 3.1-3.3**: State representation, tensor structure, normalization
   - **GitHub**: https://github.com/ZhengyaoJiang/PGPortfolio

2. **Jiang, Z., & Liang, J. (2017)**. "Cryptocurrency Portfolio Management with Deep Reinforcement Learning." arXiv:1612.01277
   - Earlier cryptocurrency-specific version

## Future Enhancements

- [ ] Support for intraday data (hourly, 30-min) for comparison
- [ ] Transaction cost modeling (realistic slippage and fees)
- [ ] Alternative normalization methods (z-score, min-max)
- [ ] Additional features (technical indicators if needed)
- [ ] Multi-resolution features (different time scales)
- [ ] Parallel processing for faster dataset generation
- [ ] Streaming data pipeline for live trading

## File Structure

```
data/
├── README.md                    # This file - comprehensive guide
├── IMPLEMENTATION_SUMMARY.md    # Week 1 implementation summary
├── TECHNICAL_INDICATORS.md      # Research analysis of all papers
├── config.py                    # Configuration classes (~90 lines)
├── data_loader.py              # Database queries (~145 lines)
├── data_processor.py           # DataFrame → tensor (~195 lines)
├── data_transformer.py         # Train/val/test splits (~623 lines)
├── schemas.sql                 # Database schema reference
├── cache/                      # Cached processed datasets
│   ├── README.md              # Cache usage documentation
│   ├── train.npz              # Training set (compressed)
│   ├── val.npz                # Validation set
│   └── test.npz               # Test set
└── __pycache__/

tests/
├── test_data_loader.py         # Database query tests
├── test_data_processor.py      # Tensor transformation tests
└── test_data_transformer.py    # Orchestration tests
```

## Contributing

When extending this module:
1. Maintain separation of concerns (loader → processor → transformer)
2. Keep functions simple and testable
3. Document design decisions with citations
4. Add tests for new functionality

## Contact

For questions about the implementation or design decisions, please refer to:
- The inline code documentation
- `TECHNICAL_INDICATORS.md` for paper summaries
- The original Jiang et al. papers
