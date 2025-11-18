"""
Portfolio action catalog for DQN.

This module defines a discrete action space for DQN by creating a catalog
of portfolio allocation strategies. Instead of discretizing each asset's
weight independently (which causes combinatorial explosion), we define a
fixed set of interpretable allocation rules.

The catalog approach solves the DQN action space problem while remaining
tractable (~30-50 discrete actions) and interpretable. Each catalog entry
represents a different allocation strategy (e.g., equal-weight top-5,
concentrated 60/40 split, diversified uniform).

Design Philosophy:
- Fixed catalog size independent of universe size A_t
- Strategies adapt dynamically to current tradable assets
- Cover diverse allocation patterns (concentrated ↔ diversified)
- All portfolios satisfy simplex constraint (sum to 1.0)

Inspired by Lucarelli & Borrotti (2020) action space discretization,
adapted for variable universe sizes in crypto portfolio management.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass


@dataclass
class Strategy:
    """
    Portfolio allocation strategy specification.
    
    Attributes
    ----------
    name : str
        Human-readable strategy identifier
    type : str
        Strategy type (e.g., 'equal_weight', 'concentrated', 'market_cap')
    params : dict
        Strategy-specific parameters
    """
    name: str
    type: str
    params: Dict[str, Any]


class PortfolioCatalog:
    """
    Catalog of discrete portfolio allocation strategies for DQN.
    
    Provides a fixed-size action space where each action corresponds to
    a portfolio allocation rule. Strategies are applied dynamically to
    the current tradable universe, handling variable A_t gracefully.
    
    Parameters
    ----------
    max_assets : int, optional
        Maximum expected assets in universe (for validation)
    
    Attributes
    ----------
    strategies : List[Strategy]
        Ordered list of allocation strategies
    size : int
        Number of discrete actions (catalog size)
    
    Examples
    --------
    >>> catalog = PortfolioCatalog()
    >>> print(f"Action space size: {catalog.size}")
    Action space size: 48
    
    >>> # Apply strategy to current observation
    >>> obs = {'asset_ids': ['BTC', 'ETH', 'SOL'], ...}
    >>> weights = catalog.apply_strategy(0, obs)  # Action 0
    >>> print(weights)
    [0.333, 0.333, 0.334]  # Equal weight across 3 assets
    """
    
    def __init__(self, max_assets: int = 50):
        self.max_assets = max_assets
        self.strategies = self._build_catalog()
        self.size = len(self.strategies)
    
    def _build_catalog(self) -> List[Strategy]:
        """
        Build the portfolio strategy catalog.
        
        Returns
        -------
        strategies : List[Strategy]
            Complete catalog of allocation strategies
        """
        catalog = []
        
        # ================================================================
        # EQUAL WEIGHT STRATEGIES (10 strategies)
        # ================================================================
        # Allocate equally among top K assets
        for k in [1, 2, 3, 5, 7, 10, 15, 20, 25, None]:
            name = f"EqualWeight_Top{k}" if k else "EqualWeight_All"
            catalog.append(Strategy(
                name=name,
                type="equal_weight",
                params={"top_k": k}
            ))
        
        # ================================================================
        # CONCENTRATED STRATEGIES (12 strategies)
        # ================================================================
        # Focus allocation on top few assets
        
        # Top 2 splits
        for w1, w2 in [(0.70, 0.30), (0.60, 0.40), (0.80, 0.20)]:
            catalog.append(Strategy(
                name=f"Concentrated_2_{int(w1*100)}_{int(w2*100)}",
                type="concentrated",
                params={"n": 2, "weights": [w1, w2]}
            ))
        
        # Top 3 splits
        for w1, w2, w3 in [(0.50, 0.30, 0.20), (0.40, 0.35, 0.25), (0.60, 0.25, 0.15)]:
            catalog.append(Strategy(
                name=f"Concentrated_3_{int(w1*100)}_{int(w2*100)}_{int(w3*100)}",
                type="concentrated",
                params={"n": 3, "weights": [w1, w2, w3]}
            ))
        
        # Top 4 splits
        for w1, w2, w3, w4 in [(0.40, 0.30, 0.20, 0.10), (0.35, 0.25, 0.20, 0.20)]:
            catalog.append(Strategy(
                name=f"Concentrated_4_{int(w1*100)}_{int(w2*100)}_{int(w3*100)}_{int(w4*100)}",
                type="concentrated",
                params={"n": 4, "weights": [w1, w2, w3, w4]}
            ))
        
        # Top 5 splits
        for weights in [[0.30, 0.25, 0.20, 0.15, 0.10], [0.25, 0.25, 0.20, 0.15, 0.15]]:
            w_str = "_".join(str(int(w*100)) for w in weights)
            catalog.append(Strategy(
                name=f"Concentrated_5_{w_str}",
                type="concentrated",
                params={"n": 5, "weights": weights}
            ))
        
        # Top 10 concentrated
        catalog.append(Strategy(
            name="Concentrated_10_Linear",
            type="concentrated",
            params={"n": 10, "weights": np.linspace(0.15, 0.05, 10).tolist()}
        ))
        
        # ================================================================
        # DIVERSIFIED STRATEGIES (8 strategies)
        # ================================================================
        # Spread allocation broadly
        
        # Capped equal weight (no asset > X%)
        for max_weight in [0.15, 0.20, 0.25, 0.30]:
            catalog.append(Strategy(
                name=f"Diversified_Cap{int(max_weight*100)}",
                type="diversified_capped",
                params={"max_weight": max_weight}
            ))
        
        # Tiered allocations
        catalog.append(Strategy(
            name="Diversified_Tiered_3Levels",
            type="diversified_tiered",
            params={"tier_sizes": [3, 5, None], "tier_weights": [0.15, 0.10, 0.05]}
        ))
        
        # Inverse rank weighting
        for decay in [0.5, 0.7, 0.9]:
            catalog.append(Strategy(
                name=f"Diversified_InverseRank_Decay{int(decay*10)}",
                type="inverse_rank",
                params={"decay": decay}
            ))
        
        # ================================================================
        # MARKET-CAP WEIGHTED STRATEGIES (8 strategies)
        # ================================================================
        # Weight by market cap (if available in obs)
        
        # Pure market cap
        catalog.append(Strategy(
            name="MarketCap_Pure",
            type="market_cap",
            params={"power": 1.0, "max_weight": 1.0}
        ))
        
        # Square root of market cap (less concentrated)
        catalog.append(Strategy(
            name="MarketCap_Sqrt",
            type="market_cap",
            params={"power": 0.5, "max_weight": 1.0}
        ))
        
        # Market cap with concentration caps
        for max_weight in [0.20, 0.30, 0.40]:
            catalog.append(Strategy(
                name=f"MarketCap_Cap{int(max_weight*100)}",
                type="market_cap",
                params={"power": 1.0, "max_weight": max_weight}
            ))
        
        # Market cap on top K only
        for k in [5, 10, 15]:
            catalog.append(Strategy(
                name=f"MarketCap_Top{k}",
                type="market_cap",
                params={"power": 1.0, "max_weight": 1.0, "top_k": k}
            ))
        
        # ================================================================
        # VOLATILITY-BASED STRATEGIES (10 strategies)
        # ================================================================
        # Weight by inverse volatility (risk parity style)
        
        # Pure inverse volatility
        catalog.append(Strategy(
            name="InverseVol_Pure",
            type="inverse_vol",
            params={"lookback": 30, "max_weight": 1.0}
        ))
        
        # Inverse volatility with caps
        for max_weight in [0.20, 0.30, 0.40]:
            catalog.append(Strategy(
                name=f"InverseVol_Cap{int(max_weight*100)}",
                type="inverse_vol",
                params={"lookback": 30, "max_weight": max_weight}
            ))
        
        # Different lookbacks
        for lookback in [15, 45, 60]:
            catalog.append(Strategy(
                name=f"InverseVol_Lookback{lookback}",
                type="inverse_vol",
                params={"lookback": lookback, "max_weight": 0.30}
            ))
        
        # Inverse volatility on top K
        for k in [5, 10, 15]:
            catalog.append(Strategy(
                name=f"InverseVol_Top{k}",
                type="inverse_vol",
                params={"lookback": 30, "max_weight": 1.0, "top_k": k}
            ))
        
        return catalog
    
    def apply_strategy(self, action_idx: int, obs: Dict[str, Any]) -> npt.NDArray[np.float32]:
        """
        Apply a catalog strategy to generate portfolio weights.
        
        Parameters
        ----------
        action_idx : int
            Index into catalog (0 to size-1)
        obs : dict
            Environment observation with keys:
            - 'asset_ids': List of tradable assets
            - 'features': [A_t, 4, 60] OHLCV tensor (for vol/momentum)
            - 'prev_weights': [A_t] previous allocation (optional)
        
        Returns
        -------
        weights : np.ndarray
            Portfolio weights [A_t] summing to 1.0
        
        Raises
        ------
        ValueError
            If action_idx is out of bounds
        """
        if not 0 <= action_idx < self.size:
            raise ValueError(f"action_idx {action_idx} out of bounds [0, {self.size})")
        
        strategy = self.strategies[action_idx]
        A_t = len(obs['asset_ids'])
        
        # Delegate to strategy-specific handler
        if strategy.type == "equal_weight":
            return self._apply_equal_weight(strategy.params, A_t)
        elif strategy.type == "concentrated":
            return self._apply_concentrated(strategy.params, A_t)
        elif strategy.type == "diversified_capped":
            return self._apply_diversified_capped(strategy.params, A_t)
        elif strategy.type == "diversified_tiered":
            return self._apply_diversified_tiered(strategy.params, A_t)
        elif strategy.type == "inverse_rank":
            return self._apply_inverse_rank(strategy.params, A_t)
        elif strategy.type == "market_cap":
            return self._apply_market_cap(strategy.params, obs)
        elif strategy.type == "inverse_vol":
            return self._apply_inverse_vol(strategy.params, obs)
        else:
            raise ValueError(f"Unknown strategy type: {strategy.type}")
    
    # ====================================================================
    # STRATEGY IMPLEMENTATIONS
    # ====================================================================
    
    def _apply_equal_weight(self, params: Dict, A_t: int) -> npt.NDArray[np.float32]:
        """Equal weight among top K assets."""
        top_k = params['top_k']
        k = min(top_k, A_t) if top_k else A_t
        
        weights = np.zeros(A_t, dtype=np.float32)
        weights[:k] = 1.0 / k
        return weights
    
    def _apply_concentrated(self, params: Dict, A_t: int) -> npt.NDArray[np.float32]:
        """Concentrated allocation on top N assets."""
        n = min(params['n'], A_t)
        target_weights = np.array(params['weights'][:n], dtype=np.float32)
        
        weights = np.zeros(A_t, dtype=np.float32)
        weights[:n] = target_weights
        
        # Renormalize if we had to truncate
        weights = weights / weights.sum()
        return weights
    
    def _apply_diversified_capped(self, params: Dict, A_t: int) -> npt.NDArray[np.float32]:
        """Equal weight with max concentration cap."""
        max_weight = params['max_weight']
        
        # Start with equal weight
        target_weight = 1.0 / A_t
        
        if target_weight <= max_weight:
            # No cap needed
            return np.full(A_t, target_weight, dtype=np.float32)
        else:
            # Need to cap: allocate max_weight to some, rest to others
            n_at_cap = int(1.0 / max_weight)
            n_below_cap = A_t - n_at_cap
            
            if n_below_cap > 0:
                remainder = 1.0 - (n_at_cap * max_weight)
                below_cap_weight = remainder / n_below_cap
            else:
                # All at cap, just normalize
                return np.full(A_t, 1.0 / A_t, dtype=np.float32)
            
            weights = np.zeros(A_t, dtype=np.float32)
            weights[:n_at_cap] = max_weight
            weights[n_at_cap:] = below_cap_weight
            return weights
    
    def _apply_diversified_tiered(self, params: Dict, A_t: int) -> npt.NDArray[np.float32]:
        """Tiered allocation (e.g., top 3 get 15%, next 5 get 10%, rest get 5%)."""
        tier_sizes = params['tier_sizes']
        tier_weights = params['tier_weights']
        
        weights = np.zeros(A_t, dtype=np.float32)
        idx = 0
        
        for tier_size, tier_weight in zip(tier_sizes, tier_weights):
            if tier_size is None:
                # Remaining assets
                tier_size = A_t - idx
            
            end_idx = min(idx + tier_size, A_t)
            if end_idx > idx:
                weights[idx:end_idx] = tier_weight
            idx = end_idx
            
            if idx >= A_t:
                break
        
        # Normalize
        return weights / weights.sum() if weights.sum() > 0 else weights
    
    def _apply_inverse_rank(self, params: Dict, A_t: int) -> npt.NDArray[np.float32]:
        """Weight inversely proportional to rank with decay."""
        decay = params['decay']
        
        # Weights = 1 / (rank ^ decay)
        ranks = np.arange(1, A_t + 1, dtype=np.float32)
        weights = 1.0 / (ranks ** decay)
        
        # Normalize
        return weights / weights.sum()
    
    def _apply_market_cap(self, params: Dict, obs: Dict) -> npt.NDArray[np.float32]:
        """
        Weight by market cap (if available).
        
        Note: This is a placeholder. If market cap data is not in obs,
        falls back to equal weight. In production, you'd extract market cap
        from obs['features'] or a separate market cap array.
        """
        A_t = len(obs['asset_ids'])
        power = params['power']
        max_weight = params['max_weight']
        top_k = params.get('top_k', None)
        
        # Placeholder: use equal weight as proxy (TODO: integrate real market cap)
        # In real implementation: market_caps = get_market_caps(obs)
        market_caps = np.ones(A_t, dtype=np.float32)  # Placeholder
        
        # Apply power transform
        weights = market_caps ** power
        
        # Apply top_k filter if specified
        if top_k is not None:
            k = min(top_k, A_t)
            mask = np.zeros(A_t, dtype=bool)
            mask[:k] = True
            weights[~mask] = 0.0
        
        # Normalize
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(A_t, dtype=np.float32) / A_t
        
        # Apply max weight cap
        if max_weight < 1.0:
            weights = self._apply_max_weight_cap(weights, max_weight)
        
        return weights
    
    def _apply_inverse_vol(self, params: Dict, obs: Dict) -> npt.NDArray[np.float32]:
        """
        Weight by inverse volatility (risk parity).
        
        Computes volatility from features tensor using specified lookback.
        """
        A_t = len(obs['asset_ids'])
        lookback = params['lookback']
        max_weight = params['max_weight']
        top_k = params.get('top_k', None)
        
        features = obs['features']  # [A_t, 4, 60]
        close_prices = features[:, 0, :]  # [A_t, 60]
        
        # Compute returns over lookback
        lookback_window = min(lookback, close_prices.shape[1] - 1)
        returns = np.diff(close_prices[:, -lookback_window-1:], axis=1)  # [A_t, lookback]
        
        # Compute volatility (std dev of returns)
        vols = np.std(returns, axis=1)  # [A_t]
        
        # Inverse volatility weighting
        # Add small epsilon to avoid division by zero
        inv_vols = 1.0 / (vols + 1e-8)
        
        # Apply top_k filter if specified
        if top_k is not None:
            k = min(top_k, A_t)
            top_k_idx = np.argsort(inv_vols)[-k:]
            mask = np.zeros(A_t, dtype=bool)
            mask[top_k_idx] = True
            inv_vols[~mask] = 0.0
        
        # Normalize
        if inv_vols.sum() > 0:
            weights = inv_vols / inv_vols.sum()
        else:
            weights = np.ones(A_t, dtype=np.float32) / A_t
        
        # Apply max weight cap
        if max_weight < 1.0:
            weights = self._apply_max_weight_cap(weights, max_weight)
        
        return weights.astype(np.float32)
    
    def _apply_max_weight_cap(self, weights: npt.NDArray, max_weight: float, 
                              max_iterations: int = 10) -> npt.NDArray:
        """
        Apply maximum weight constraint via iterative redistribution.
        
        Parameters
        ----------
        weights : np.ndarray
            Initial weights [A_t]
        max_weight : float
            Maximum allowed weight per asset
        max_iterations : int
            Maximum redistribution iterations
        
        Returns
        -------
        capped_weights : np.ndarray
            Weights with max constraint enforced
        """
        weights = weights.copy()
        
        for _ in range(max_iterations):
            # Find assets exceeding cap
            over_cap = weights > max_weight
            
            if not over_cap.any():
                break
            
            # Cap those assets
            excess = (weights[over_cap] - max_weight).sum()
            weights[over_cap] = max_weight
            
            # Redistribute excess to assets below cap
            below_cap = ~over_cap
            if below_cap.any():
                redistribution = excess / below_cap.sum()
                weights[below_cap] += redistribution
        
        # Final normalization
        return weights / weights.sum()
    
    def get_strategy_name(self, action_idx: int) -> str:
        """Get human-readable name for a catalog action."""
        if not 0 <= action_idx < self.size:
            raise ValueError(f"action_idx {action_idx} out of bounds")
        return self.strategies[action_idx].name
    
    def __repr__(self) -> str:
        return f"PortfolioCatalog(size={self.size}, strategies={self.size})"
