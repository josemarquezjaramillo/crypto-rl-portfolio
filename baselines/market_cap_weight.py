"""
Market Cap Weighted Baseline Portfolio Strategy.

This baseline allocates weights proportional to market capitalization,
using the pre-computed index weights from the database. It matches the
methodology used by major crypto indices (e.g., CoinDesk 20, Bloomberg
Galaxy Crypto Index).

The strategy:
1. Loads monthly index constituent weights from PostgreSQL
2. Maps weights to the current tradable universe
3. Renormalizes if some assets are missing
4. Environment handles turnover/concentration constraints

Database Table: index_monthly_constituents
- period_start_date: First day of month
- coin_id: Asset identifier
- initial_market_cap_at_rebalance: Raw market cap
- initial_weight_at_rebalance: Pre-computed capped weights (35% max)
"""

import os
from dataclasses import dataclass
from typing import Dict, Optional, List
from datetime import datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from sqlalchemy import create_engine, text

from baselines.base_baseline import BaselineAgent, BaselineConfig
from environment.environment import Obs


@dataclass
class MarketCapWeightConfig(BaselineConfig):
    """
    Configuration for Market Cap Weight baseline.
    
    Attributes
    ----------
    use_sqrt_weights : bool
        If True, use sqrt(market_cap) for less concentrated weights.
        Default False uses raw market cap weights.
    max_weight_per_asset : float, optional
        Per-asset cap (e.g., 0.35 for 35% max). If None, uses database weights
        which already have caps applied.
    db_host : str
        Database host
    db_port : int
        Database port
    db_name : str
        Database name
    db_user : str
        Database username
    db_password : str
        Database password
    """
    use_sqrt_weights: bool = False
    max_weight_per_asset: Optional[float] = None
    db_host: str = None
    db_port: int = None
    db_name: str = None
    db_user: str = None
    db_password: str = None
    
    def __init__(
        self,
        random_seed: int = 42,
        log_dir: Optional[Path] = None,
        use_sqrt_weights: bool = False,
        max_weight_per_asset: Optional[float] = None,
        db_host: str = None,
        db_port: int = None,
        db_name: str = None,
        db_user: str = None,
        db_password: str = None,
    ):
        # Initialize parent class
        super().__init__(
            name="MarketCapWeight",
            random_seed=random_seed,
            log_dir=log_dir,
        )
        # Set additional attributes
        self.use_sqrt_weights = use_sqrt_weights
        self.max_weight_per_asset = max_weight_per_asset
        
        # Database config from env vars if not provided
        self.db_host = db_host or os.getenv('DB_HOST', 'localhost')
        self.db_port = db_port or int(os.getenv('DB_PORT', '5432'))
        self.db_name = db_name or os.getenv('DB_NAME')
        self.db_user = db_user or os.getenv('DB_USER', 'postgres')
        self.db_password = db_password or os.getenv('DB_PASSWORD', '')


class MarketCapWeightAgent(BaselineAgent):
    """
    Market-cap weighted portfolio baseline.
    
    Allocates weights proportional to market capitalization using monthly
    index constituent data from the database. This mirrors how major
    cryptocurrency indices operate.
    
    Weight Calculation:
    1. Load constituent weights for the current month from database
    2. Filter to assets in current tradable universe
    3. Renormalize to sum to 1.0
    4. Optionally apply sqrt transformation for less concentration
    5. Environment projects to satisfy turnover/concentration constraints
    
    The database stores pre-computed weights with a 35% cap already applied,
    but additional caps can be configured via max_weight_per_asset.
    
    Example
    -------
    >>> from baselines import MarketCapWeightAgent
    >>> from baselines.market_cap_weight import MarketCapWeightConfig
    >>> 
    >>> config = MarketCapWeightConfig(
    ...     random_seed=42,
    ...     use_sqrt_weights=False,  # Use raw market cap
    ... )
    >>> agent = MarketCapWeightAgent(config)
    >>> 
    >>> # Run evaluation
    >>> results = agent.evaluate(env, n_episodes=1)
    >>> print(f"Return: {results['mean_return']:.4f}")
    """
    
    def __init__(self, config: MarketCapWeightConfig = None):
        """
        Initialize Market Cap Weight agent.
        
        Parameters
        ----------
        config : MarketCapWeightConfig, optional
            Agent configuration. If None, uses defaults with env vars.
        """
        if config is None:
            config = MarketCapWeightConfig()
        super().__init__(config)
        
        self.config: MarketCapWeightConfig = config
        
        # Database connection
        self._engine = None
        self._init_database()
        
        # Cache for monthly weights (avoid repeated DB queries)
        # Key: "YYYY-MM" -> Dict[coin_id, weight]
        self._weight_cache: Dict[str, Dict[str, float]] = {}
    
    def _init_database(self):
        """Initialize database connection."""
        conn_string = (
            f"postgresql://{self.config.db_user}:{self.config.db_password}"
            f"@{self.config.db_host}:{self.config.db_port}/{self.config.db_name}"
        )
        self._engine = create_engine(conn_string)
    
    def _get_month_key(self, date: np.datetime64) -> str:
        """Convert datetime64 to month key 'YYYY-MM'."""
        # Convert to pandas timestamp for easier manipulation
        ts = pd.Timestamp(date)
        return f"{ts.year:04d}-{ts.month:02d}"
    
    def _get_period_start_date(self, date: np.datetime64) -> str:
        """Get the first day of the month as 'YYYY-MM-DD'."""
        ts = pd.Timestamp(date)
        return f"{ts.year:04d}-{ts.month:02d}-01"
    
    def _load_monthly_weights(self, period_date: str) -> Dict[str, float]:
        """
        Load market cap weights from database for a given month.
        
        Parameters
        ----------
        period_date : str
            First day of month 'YYYY-MM-DD'
            
        Returns
        -------
        weights : Dict[str, float]
            Mapping coin_id -> weight
        """
        query = text("""
            SELECT coin_id, initial_market_cap_at_rebalance, initial_weight_at_rebalance
            FROM index_monthly_constituents
            WHERE period_start_date = :period_date
            ORDER BY initial_weight_at_rebalance DESC
        """)
        
        with self._engine.connect() as conn:
            result = conn.execute(query, {'period_date': period_date})
            rows = result.fetchall()
        
        if not rows:
            return {}
        
        weights = {}
        
        if self.config.use_sqrt_weights:
            # Use sqrt of market cap for less concentrated weights
            total_sqrt_mcap = 0.0
            sqrt_mcaps = {}
            
            for row in rows:
                coin_id = row[0]
                mcap = float(row[1])  # initial_market_cap_at_rebalance
                sqrt_mcap = np.sqrt(mcap)
                sqrt_mcaps[coin_id] = sqrt_mcap
                total_sqrt_mcap += sqrt_mcap
            
            if total_sqrt_mcap > 0:
                for coin_id, sqrt_mcap in sqrt_mcaps.items():
                    weights[coin_id] = sqrt_mcap / total_sqrt_mcap
        else:
            # Use pre-computed weights from database (already capped at 35%)
            for row in rows:
                coin_id = row[0]
                weight = float(row[2])  # initial_weight_at_rebalance
                weights[coin_id] = weight
        
        return weights
    
    def _apply_concentration_cap(
        self, 
        weights: Dict[str, float], 
        max_weight: float
    ) -> Dict[str, float]:
        """
        Apply concentration cap and renormalize.
        
        Iteratively caps weights and redistributes excess to uncapped assets.
        
        Parameters
        ----------
        weights : Dict[str, float]
            Input weights (may sum to != 1.0)
        max_weight : float
            Maximum weight per asset
            
        Returns
        -------
        capped_weights : Dict[str, float]
            Capped and normalized weights
        """
        # Normalize first
        total = sum(weights.values())
        if total <= 0:
            return weights
        
        weights = {k: v / total for k, v in weights.items()}
        
        # Iteratively cap and redistribute
        for _ in range(10):  # Max iterations to prevent infinite loop
            excess = 0.0
            capped_coins = set()
            uncapped_total = 0.0
            
            # Find excess and uncapped total
            for coin, w in weights.items():
                if w > max_weight:
                    excess += w - max_weight
                    capped_coins.add(coin)
                else:
                    uncapped_total += w
            
            if excess < 1e-9:
                break
            
            # Apply cap and redistribute
            new_weights = {}
            for coin, w in weights.items():
                if coin in capped_coins:
                    new_weights[coin] = max_weight
                elif uncapped_total > 0:
                    # Proportionally redistribute excess
                    new_weights[coin] = w + excess * (w / uncapped_total)
                else:
                    new_weights[coin] = w
            
            weights = new_weights
        
        # Final normalization
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    def select_action(self, obs: Obs) -> npt.NDArray[np.float32]:
        """
        Select market-cap weighted allocation.
        
        Parameters
        ----------
        obs : Obs
            Current observation with asset_ids and date
            
        Returns
        -------
        weights : np.ndarray
            Market-cap proportional weights for tradable assets
        """
        asset_ids = obs['asset_ids']
        date = obs['date']
        n_assets = len(asset_ids)
        
        if n_assets == 0:
            return np.array([], dtype=np.float32)
        
        # Get month key for caching
        month_key = self._get_month_key(date)
        
        # Load weights from cache or database
        if month_key not in self._weight_cache:
            period_date = self._get_period_start_date(date)
            self._weight_cache[month_key] = self._load_monthly_weights(period_date)
        
        monthly_weights = self._weight_cache[month_key]
        
        # Map to current tradable universe
        weights_dict = {}
        for asset_id in asset_ids:
            if asset_id in monthly_weights:
                weights_dict[asset_id] = monthly_weights[asset_id]
            else:
                # Asset not in index - assign zero (will be redistributed)
                weights_dict[asset_id] = 0.0
        
        # Renormalize to sum to 1.0
        total = sum(weights_dict.values())
        if total > 0:
            weights_dict = {k: v / total for k, v in weights_dict.items()}
        else:
            # Fallback to equal weight if no weights found
            weights_dict = {k: 1.0 / n_assets for k in asset_ids}
        
        # Apply additional concentration cap if configured
        if self.config.max_weight_per_asset is not None:
            weights_dict = self._apply_concentration_cap(
                weights_dict, 
                self.config.max_weight_per_asset
            )
        
        # Convert to array in asset_ids order
        weights = np.array(
            [weights_dict.get(aid, 0.0) for aid in asset_ids],
            dtype=np.float32
        )
        
        # Final normalization (safety check)
        total = weights.sum()
        if total > 0:
            weights = weights / total
        else:
            weights = np.full(n_assets, 1.0 / n_assets, dtype=np.float32)
        
        return weights
    
    def get_name(self) -> str:
        """Return strategy name."""
        if self.config.use_sqrt_weights:
            return "Market Cap Weight (sqrt)"
        return "Market Cap Weight"
    
    def clear_cache(self):
        """Clear the weight cache (useful for testing)."""
        self._weight_cache.clear()
