"""
DatasetBackend: Adapter for ExportedDataset to PortfolioEnv interface.

This module provides a concrete implementation of the DataBackend interface
expected by PortfolioEnv, wrapping the ExportedDataset class from dataset_loader.py.

Key responsibilities:
- Type conversion: string dates ↔ np.datetime64
- Interface adaptation: dict → tuple
- Split tag filtering: enable train/val/test subset selection
- Leakage-safe data access: no look-ahead information

Example Usage
-------------
>>> from pathlib import Path
>>> from data.dataset_loader import load_exported_dataset
>>> from data.dataset_backend import DatasetBackend
>>> from environment.environment import PortfolioEnv, EnvConfig
>>>
>>> # Load the dev split from disk
>>> ds = load_exported_dataset("dataset_v1", split="dev")
>>>
>>> # Create backend for training (only train_core days)
>>> train_backend = DatasetBackend(ds, split_tag_filter="train_core")
>>> print(f"Training days: {len(train_backend.dates())}")  # 922 days
>>>
>>> # Create backend for a specific validation window
>>> val_backend = DatasetBackend(ds, split_tag_filter="val_window_val_bear")
>>> print(f"Validation days: {len(val_backend.dates())}")  # 19 days
>>>
>>> # Create backend for multiple validation windows
>>> val_multi = DatasetBackend(
...     ds,
...     split_tag_filter=["val_window_val_bear", "val_window_val_chop"]
... )
>>> print(f"Combined val days: {len(val_multi.dates())}")  # 38 days
>>>
>>> # Use with environment
>>> cfg = EnvConfig(split="train", cost_rate=0.001, turnover_cap=0.30)
>>> env = PortfolioEnv(cfg, train_backend)
>>> obs = env.reset()
>>> action = env.sample_action()
>>> obs_next, reward, done, info = env.step(action)
"""

from typing import List, Optional, Tuple, Dict, Any, Union
import numpy as np
import numpy.typing as npt

from data.dataset_loader import ExportedDataset


class DatasetBackend:
    """
    Adapter that wraps ExportedDataset and implements the DataBackend interface
    expected by PortfolioEnv.
    
    This class bridges the gap between the data loading layer (dataset_loader.py)
    and the environment layer (environment.py) by:
    
    1. Converting string dates to np.datetime64 objects
    2. Adapting dict return values to tuple interface
    3. Filtering dates by split_tag (e.g., train_core, validation windows)
    4. Providing metadata access for environment configuration
    
    The adapter maintains chronological ordering of dates and ensures no data
    leakage by only exposing precomputed, leakage-safe tensors.
    
    Parameters
    ----------
    exported_dataset : ExportedDataset
        A loaded dataset from load_exported_dataset(). Should be either
        "dev" or "test" split.
    split_tag_filter : str | List[str] | None, optional
        Filter to apply to the split_tag column:
        - None: use all dates in the split (default)
        - "train_core": only training days (e.g., 922 days in dev)
        - "val_window_val_bear": specific validation window
        - ["tag1", "tag2"]: multiple tags combined
        
    Raises
    ------
    ValueError
        If split_tag_filter results in empty dataset or contains invalid tags.
        
    Attributes
    ----------
    metadata : Dict[str, Any]
        Access to underlying dataset metadata (lookback, turnover_cap, etc.)
    
    Examples
    --------
    Load dev split and filter for training:
    
    >>> ds = load_exported_dataset("dataset_v1", split="dev")
    >>> backend = DatasetBackend(ds, split_tag_filter="train_core")
    >>> dates = backend.dates()
    >>> print(f"Shape: {dates.shape}, dtype: {dates.dtype}")
    Shape: (922,), dtype: datetime64[D]
    
    Fetch data for a specific date:
    
    >>> first_date = dates[0]
    >>> features, asset_ids, fwd_returns = backend.get_day(first_date)
    >>> print(f"Assets: {len(asset_ids)}, Features: {features.shape}")
    Assets: 10, Features: (10, 4, 60)
    """
    
    def __init__(
        self,
        exported_dataset: ExportedDataset,
        split_tag_filter: Optional[Union[str, List[str]]] = None,
    ):
        """
        Initialize the backend adapter with optional split tag filtering.
        
        The constructor performs filtering, builds bidirectional date conversion
        mappings, and validates that the resulting dataset is non-empty.
        """
        self._ds = exported_dataset
        self._split_tag_filter = split_tag_filter
        
        # Filter dates by split_tag if requested
        self._valid_dates = self._apply_split_tag_filter()
        
        # Validate non-empty
        if len(self._valid_dates) == 0:
            raise ValueError(
                f"Filtered dataset is empty. "
                f"split_tag_filter={split_tag_filter}, "
                f"available tags: {sorted(self._ds.index_df['split_tag'].unique())}"
            )
        
        # Build bidirectional conversion dicts: string ↔ np.datetime64
        # This is done once at init for efficiency
        self._date_str_to_dt64: Dict[str, np.datetime64] = {
            date_str: np.datetime64(date_str, 'D')
            for date_str in self._valid_dates
        }
        self._dt64_to_date_str: Dict[np.datetime64, str] = {
            v: k for k, v in self._date_str_to_dt64.items()
        }
        
        # Precompute the dates array (sorted chronologically)
        self._dates_array: npt.NDArray[np.datetime64] = np.array(
            [self._date_str_to_dt64[s] for s in self._valid_dates],
            dtype='datetime64[D]'
        )
    
    def _apply_split_tag_filter(self) -> List[str]:
        """
        Filter dates based on split_tag_filter parameter.
        
        Returns
        -------
        List[str]
            Filtered date strings in chronological order.
        """
        # Case 1: No filter → use all dates
        if self._split_tag_filter is None:
            return self._ds.dates()
        
        # Case 2: Single tag string
        if isinstance(self._split_tag_filter, str):
            # Validate tag exists
            available_tags = set(self._ds.index_df['split_tag'].unique())
            if self._split_tag_filter not in available_tags:
                raise ValueError(
                    f"Split tag '{self._split_tag_filter}' not found in dataset. "
                    f"Available tags: {sorted(available_tags)}"
                )
            
            # Filter index_df
            mask = self._ds.index_df['split_tag'] == self._split_tag_filter
            filtered_df = self._ds.index_df[mask]
            return [d.strftime('%Y-%m-%d') for d in filtered_df['date']]
        
        # Case 3: List of tags
        if isinstance(self._split_tag_filter, list):
            # Validate all tags exist
            available_tags = set(self._ds.index_df['split_tag'].unique())
            invalid_tags = set(self._split_tag_filter) - available_tags
            if invalid_tags:
                raise ValueError(
                    f"Invalid split tags: {sorted(invalid_tags)}. "
                    f"Available tags: {sorted(available_tags)}"
                )
            
            # Filter with .isin() for multiple tags
            mask = self._ds.index_df['split_tag'].isin(self._split_tag_filter)
            filtered_df = self._ds.index_df[mask]
            return [d.strftime('%Y-%m-%d') for d in filtered_df['date']]
        
        # Invalid type
        raise TypeError(
            f"split_tag_filter must be str, List[str], or None, "
            f"got {type(self._split_tag_filter)}"
        )
    
    def dates(self) -> npt.NDArray[np.datetime64]:
        """
        Return ordered decision dates for the filtered split.
        
        Returns dates as a 1D numpy array of datetime64[D] (day precision).
        Dates are guaranteed to be in chronological order.
        
        Returns
        -------
        np.ndarray
            Shape (N,) where N is the number of dates after filtering.
            Dtype is datetime64[D] (day precision, timezone-naive).
            
        Examples
        --------
        >>> backend = DatasetBackend(ds, split_tag_filter="train_core")
        >>> dates = backend.dates()
        >>> print(dates[0], dates[-1])
        2018-09-01 2023-12-31
        >>> print(dates.dtype)
        datetime64[D]
        """
        return self._dates_array.copy()
    
    def get_day(
        self,
        date: np.datetime64
    ) -> Tuple[
        npt.NDArray[np.float32],  # features: [A_t, 4, 60]
        List[str],                # asset_ids
        npt.NDArray[np.float32],  # fwd_returns: [A_t]
    ]:
        """
        Fetch precomputed data for a specific decision date.
        
        This method is the core interface used by PortfolioEnv. It returns:
        1. Features: [A_t, 4, 60] normalized OHLCV lookback tensor
        2. Asset IDs: List of asset identifiers aligned to feature rows
        3. Forward returns: [A_t] simple returns from t → t+1
        
        The features are lagged (no look-ahead), while forward returns are
        precomputed for reward calculation only (not exposed to the agent).
        
        Parameters
        ----------
        date : np.datetime64
            Decision date to fetch. Must be a date in the filtered dataset.
            
        Returns
        -------
        features : np.ndarray
            Shape [A_t, 4, 60], dtype float32.
            Channels: [Close, High, Low, Volume]
            Temporal: 60-day lookback window up to and including `date`
        asset_ids : List[str]
            Asset identifiers, length A_t. Aligned to features rows.
        fwd_returns : np.ndarray
            Shape [A_t], dtype float32.
            Simple returns from date t → t+1 for each asset.
            
        Raises
        ------
        KeyError
            If the date is not in the filtered dataset.
            
        Examples
        --------
        >>> first_date = backend.dates()[0]
        >>> features, asset_ids, fwd_returns = backend.get_day(first_date)
        >>> print(f"Date: {first_date}")
        Date: 2018-09-01
        >>> print(f"Assets: {asset_ids[:3]}")
        Assets: ['bitcoin', 'ethereum', 'ripple']
        >>> print(f"Features shape: {features.shape}")
        Features shape: (10, 4, 60)
        >>> print(f"Forward returns: {fwd_returns.shape}")
        Forward returns: (10,)
        """
        # Convert np.datetime64 → string for dict lookup
        date_str = self._dt64_to_date_str.get(date)
        
        if date_str is None:
            raise KeyError(
                f"Date {date} not in filtered dataset. "
                f"Applied filter: {self._split_tag_filter}. "
                f"Available date range: {self._dates_array[0]} to {self._dates_array[-1]}"
            )
        
        # Fetch from underlying ExportedDataset
        day_data = self._ds.get_day(date_str)
        
        # Extract and return as tuple (interface requirement)
        return (
            day_data["obs_tensor"],
            day_data["assets"],
            day_data["fwd_returns"],
        )
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """
        Access underlying dataset metadata.
        
        Provides access to configuration parameters like lookback_days,
        turnover_cap, validation windows, etc. from metadata.json.
        
        Returns
        -------
        Dict[str, Any]
            Metadata dictionary with keys like:
            - lookback_days: int (e.g., 60)
            - turnover_cap_l1: float (e.g., 0.3)
            - dev_start_date, dev_end_date: str
            - validation_windows: List[Dict]
            - etc.
            
        Examples
        --------
        >>> backend.metadata['lookback_days']
        60
        >>> backend.metadata['turnover_cap_l1']
        0.3
        """
        return self._ds.metadata
    
    def get_split_tag(self, date: np.datetime64) -> str:
        """
        Get the split_tag for a specific date.
        
        Useful for logging and debugging to identify whether a date is
        from train_core, a validation window, or test set.
        
        Parameters
        ----------
        date : np.datetime64
            Date to query.
            
        Returns
        -------
        str
            Split tag, e.g., "train_core", "val_window_val_bear", "test".
            
        Raises
        ------
        KeyError
            If the date is not in the filtered dataset.
            
        Examples
        --------
        >>> first_date = backend.dates()[0]
        >>> backend.get_split_tag(first_date)
        'train_core'
        """
        date_str = self._dt64_to_date_str.get(date)
        
        if date_str is None:
            raise KeyError(f"Date {date} not in filtered dataset")
        
        day_data = self._ds.get_day(date_str)
        return day_data["split_tag"]
    
    def __len__(self) -> int:
        """Return number of dates in filtered dataset."""
        return len(self._valid_dates)
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"DatasetBackend("
            f"split={self._ds.split}, "
            f"filter={self._split_tag_filter}, "
            f"dates={len(self._valid_dates)})"
        )


if __name__ == "__main__":
    """
    Self-test and usage examples.
    
    Run with:
        cd /home/jlmarquez11/crypto-rl-portfolio
        python -m data.dataset_backend
    """
    from pathlib import Path
    from data.dataset_loader import load_exported_dataset
    
    print("=" * 70)
    print("DatasetBackend Self-Test")
    print("=" * 70)
    
    # Test 1: Load dev split and create training backend
    print("\n[Test 1] Load dev split and filter for training")
    ds = load_exported_dataset("dataset_v1", split="dev")
    backend = DatasetBackend(ds, split_tag_filter="train_core")
    print(f"✓ Created backend: {backend}")
    print(f"  Training days: {len(backend)}")
    
    # Test 2: Check dates type and properties
    print("\n[Test 2] Check dates() return type")
    dates = backend.dates()
    assert isinstance(dates, np.ndarray), "dates() should return np.ndarray"
    assert dates.dtype == np.dtype('datetime64[D]'), f"Expected datetime64[D], got {dates.dtype}"
    print(f"✓ Dates array: shape={dates.shape}, dtype={dates.dtype}")
    print(f"  Date range: {dates[0]} to {dates[-1]}")
    
    # Test 3: Fetch a day and validate structure
    print("\n[Test 3] Fetch data for first training day")
    first_date = dates[0]
    features, asset_ids, fwd_returns = backend.get_day(first_date)
    assert features.shape[1:] == (4, 60), f"Expected (*, 4, 60), got {features.shape}"
    assert len(asset_ids) == features.shape[0], "Asset IDs length mismatch"
    assert len(fwd_returns) == features.shape[0], "Forward returns length mismatch"
    assert features.dtype == np.float32, f"Expected float32, got {features.dtype}"
    assert fwd_returns.dtype == np.float32, f"Expected float32, got {fwd_returns.dtype}"
    print(f"✓ Day {first_date}:")
    print(f"  Assets: {len(asset_ids)} (sample: {asset_ids[:3]})")
    print(f"  Features shape: {features.shape}")
    print(f"  Forward returns shape: {fwd_returns.shape}")
    
    # Test 4: Get split tag
    print("\n[Test 4] Get split tag for a date")
    tag = backend.get_split_tag(first_date)
    assert tag == "train_core", f"Expected 'train_core', got '{tag}'"
    print(f"✓ Split tag: '{tag}'")
    
    # Test 5: Multiple validation windows
    print("\n[Test 5] Filter for multiple validation windows")
    val_backend = DatasetBackend(
        ds,
        split_tag_filter=["val_window_val_bear", "val_window_val_chop"]
    )
    val_dates = val_backend.dates()
    print(f"✓ Created validation backend: {val_backend}")
    print(f"  Combined validation days: {len(val_dates)}")
    
    # Test 6: No filter (all dev dates)
    print("\n[Test 6] No filter (all dev dates)")
    all_backend = DatasetBackend(ds, split_tag_filter=None)
    all_dates = all_backend.dates()
    print(f"✓ Created unfiltered backend: {all_backend}")
    print(f"  Total dev days: {len(all_dates)}")
    
    # Test 7: Test split filtering
    print("\n[Test 7] Load test split")
    test_ds = load_exported_dataset("dataset_v1", split="test")
    test_backend = DatasetBackend(test_ds, split_tag_filter=None)
    test_dates = test_backend.dates()
    print(f"✓ Created test backend: {test_backend}")
    print(f"  Test days: {len(test_dates)}")
    print(f"  Test range: {test_dates[0]} to {test_dates[-1]}")
    
    # Test 8: Metadata access
    print("\n[Test 8] Access metadata")
    meta = backend.metadata
    print(f"✓ Metadata keys: {list(meta.keys())[:5]}...")
    print(f"  Lookback days: {meta['lookback_days']}")
    print(f"  Turnover cap: {meta['turnover_cap_l1']}")
    print(f"  Validation windows: {len(meta['validation_windows'])}")
    
    # Test 9: Error handling - invalid tag
    print("\n[Test 9] Error handling - invalid split tag")
    try:
        bad_backend = DatasetBackend(ds, split_tag_filter="nonexistent_tag")
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {str(e)[:60]}...")
    
    # Test 10: Error handling - invalid date
    print("\n[Test 10] Error handling - date not in filtered set")
    try:
        invalid_date = np.datetime64('2030-01-01', 'D')
        backend.get_day(invalid_date)
        print("✗ Should have raised KeyError")
    except KeyError as e:
        print(f"✓ Correctly raised KeyError: {str(e)[:60]}...")
    
    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)
    
    # Display usage example
    print("\n" + "=" * 70)
    print("Usage Example with PortfolioEnv")
    print("=" * 70)
    print("""
from data.dataset_loader import load_exported_dataset
from data.dataset_backend import DatasetBackend
from environment.environment import PortfolioEnv, EnvConfig

# Load and prepare data
ds = load_exported_dataset("dataset_v1", split="dev")
train_backend = DatasetBackend(ds, split_tag_filter="train_core")

# Create environment
cfg = EnvConfig(
    split="train",
    cost_rate=0.001,
    turnover_cap=0.30,
    action_mode="continuous",
)
env = PortfolioEnv(cfg, train_backend)

# Run episode
obs = env.reset()
for step in range(100):
    action = env.sample_action()
    obs, reward, done, info = env.step(action)
    if done:
        break

print(f"Episode finished after {step+1} steps")
print(f"Final portfolio value: {info['portfolio_value']:.4f}")
    """)
