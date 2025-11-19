"""
Delta-based action catalog for portfolio rebalancing.

This module implements a discrete action space where each action represents
a rebalancing decision relative to the current portfolio weights, following
Lucarelli & Borrotti (2020). Instead of selecting fixed allocation strategies,
actions adjust the existing portfolio (e.g., "increase BTC by 10%").

Key Advantages:
- Reflects realistic portfolio management (adjust positions, not pick strategies)
- Maintains portfolio continuity (smooth transitions vs. strategy jumps)
- Context-aware feasibility (same action can be safe or risky depending on state)
- Fixed action space size (70 actions, independent of universe size A_t)

Design Philosophy:
- Each action applies a delta to previous weights w_{t-1}
- Actions represent portfolio manager decisions (increase/decrease/rotate/diversify)
- Environment penalties teach which deltas are safe in which states
- Agent learns both profitability AND feasibility through experience

Inspired by Lucarelli & Borrotti (2020), adapted for variable universe sizes
and multi-agent compatibility (DQN, REINFORCE, A2C).
"""

from typing import Dict, List, Any, Optional
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass


@dataclass
class DeltaAction:
    """
    Delta-based rebalancing action specification.
    
    Attributes
    ----------
    name : str
        Human-readable action identifier
    type : str
        Action type (e.g., 'hold', 'adjust_topk', 'rotate')
    params : dict
        Action-specific parameters
    """
    name: str
    type: str
    params: Dict[str, Any]


class DeltaActionCatalog:
    """
    Catalog of delta-based portfolio rebalancing actions.
    
    Provides a fixed-size action space (70 actions) where each action
    represents an adjustment to the current portfolio. Actions are applied
    dynamically to the current weights, handling variable A_t gracefully.
    
    Action Categories (70 total):
    - Hold (1): No change
    - Adjust top-K (33): Increase/decrease exposure to top K assets
    - Rotate (22): Transfer weight between specific assets
    - Diversify (4): Move weight from concentrated → equal
    - Concentrate (4): Move weight from equal → top asset
    - Rebalance equal (3): Reset to equal weight
    - Shift to top-K (3): Zero bottom, equal weight top-K
    
    Parameters
    ----------
    max_assets : int, optional
        Maximum expected assets in universe (for validation)
    
    Attributes
    ----------
    actions : List[DeltaAction]
        Ordered list of rebalancing actions
    size : int
        Number of discrete actions (catalog size)
    
    Examples
    --------
    >>> catalog = DeltaActionCatalog()
    >>> print(f"Action space size: {catalog.size}")
    Action space size: 70
    
    >>> # Apply delta action to current portfolio
    >>> obs = {'asset_ids': ['BTC', 'ETH', 'SOL'], ...}
    >>> prev_weights = np.array([0.25, 0.20, 0.15, ...])
    >>> new_weights = catalog.apply_action(5, obs, prev_weights)
    >>> print(new_weights)
    [0.35, 0.18, 0.14, ...]  # Adjusted portfolio
    """
    
    def __init__(self, max_assets: int = 50):
        self.max_assets = max_assets
        self.actions = self._build_catalog()
        self.size = len(self.actions)
    
    def _build_catalog(self) -> List[DeltaAction]:
        """
        Build the delta-based rebalancing action catalog.
        
        Returns
        -------
        actions : List[DeltaAction]
            Complete catalog of 70 rebalancing actions
        """
        catalog = []
        
        # ================================================================
        # 1. HOLD (1 action)
        # ================================================================
        catalog.append(DeltaAction(
            name='hold',
            type='hold',
            params={}
        ))
        
        # ================================================================
        # 2. ADJUST TOP-1 (7 actions)
        # ================================================================
        # Full Lucarelli delta range for most important asset
        for delta in [-0.15, -0.10, -0.05, 0.00, +0.05, +0.10, +0.15]:
            catalog.append(DeltaAction(
                name=f'adjust_top1_{delta:+.0%}',
                type='adjust_topk',
                params={'k': 1, 'delta': delta}
            ))
        
        # ================================================================
        # 3. ADJUST TOP-2 EQUALLY (7 actions)
        # ================================================================
        # Adjust both top-2 by same amount (coordinated move)
        for delta in [-0.15, -0.10, -0.05, 0.00, +0.05, +0.10, +0.15]:
            catalog.append(DeltaAction(
                name=f'adjust_top2_{delta:+.0%}_each',
                type='adjust_topk',
                params={'k': 2, 'delta': delta}
            ))
        
        # ================================================================
        # 4. ADJUST TOP-3 EQUALLY (7 actions)
        # ================================================================
        for delta in [-0.15, -0.10, -0.05, 0.00, +0.05, +0.10, +0.15]:
            catalog.append(DeltaAction(
                name=f'adjust_top3_{delta:+.0%}_each',
                type='adjust_topk',
                params={'k': 3, 'delta': delta}
            ))
        
        # ================================================================
        # 5. ADJUST TOP-4 EQUALLY (7 actions)
        # ================================================================
        for delta in [-0.15, -0.10, -0.05, 0.00, +0.05, +0.10, +0.15]:
            catalog.append(DeltaAction(
                name=f'adjust_top4_{delta:+.0%}_each',
                type='adjust_topk',
                params={'k': 4, 'delta': delta}
            ))
        
        # ================================================================
        # 6. ADJUST TOP-5 EQUALLY (5 actions)
        # ================================================================
        # Reduced granularity for lower-ranked assets
        for delta in [-0.10, -0.05, 0.00, +0.05, +0.10]:
            catalog.append(DeltaAction(
                name=f'adjust_top5_{delta:+.0%}_each',
                type='adjust_topk',
                params={'k': 5, 'delta': delta}
            ))
        
        # ================================================================
        # 7. ROTATE TOP-1 ↔ TOP-2 (10 actions)
        # ================================================================
        # Transfer weight between #1 and #2 (common rebalancing)
        for amount in [0.05, 0.10, 0.15, 0.20, 0.25]:
            catalog.append(DeltaAction(
                name=f'rotate_1to2_{amount:.0%}',
                type='rotate',
                params={'from_idx': 0, 'to_idx': 1, 'amount': amount}
            ))
            catalog.append(DeltaAction(
                name=f'rotate_2to1_{amount:.0%}',
                type='rotate',
                params={'from_idx': 1, 'to_idx': 0, 'amount': amount}
            ))
        
        # ================================================================
        # 8. ROTATE TOP-1 ↔ TOP-3 (8 actions)
        # ================================================================
        for amount in [0.10, 0.15, 0.20, 0.25]:
            catalog.append(DeltaAction(
                name=f'rotate_1to3_{amount:.0%}',
                type='rotate',
                params={'from_idx': 0, 'to_idx': 2, 'amount': amount}
            ))
            catalog.append(DeltaAction(
                name=f'rotate_3to1_{amount:.0%}',
                type='rotate',
                params={'from_idx': 2, 'to_idx': 0, 'amount': amount}
            ))
        
        # ================================================================
        # 9. ROTATE TOP-2 ↔ TOP-3 (4 actions)
        # ================================================================
        for amount in [0.10, 0.15]:
            catalog.append(DeltaAction(
                name=f'rotate_2to3_{amount:.0%}',
                type='rotate',
                params={'from_idx': 1, 'to_idx': 2, 'amount': amount}
            ))
            catalog.append(DeltaAction(
                name=f'rotate_3to2_{amount:.0%}',
                type='rotate',
                params={'from_idx': 2, 'to_idx': 1, 'amount': amount}
            ))
        
        # ================================================================
        # 10. DIVERSIFY (4 actions)
        # ================================================================
        # Move weight from concentrated → equal distribution
        for amount in [0.05, 0.10, 0.15, 0.20]:
            catalog.append(DeltaAction(
                name=f'diversify_{amount:.0%}',
                type='diversify',
                params={'amount': amount}
            ))
        
        # ================================================================
        # 11. CONCENTRATE (4 actions)
        # ================================================================
        # Move weight from equal → top asset
        for amount in [0.05, 0.10, 0.15, 0.20]:
            catalog.append(DeltaAction(
                name=f'concentrate_{amount:.0%}',
                type='concentrate',
                params={'amount': amount}
            ))
        
        # ================================================================
        # 12. REBALANCE TO EQUAL (3 actions)
        # ================================================================
        # Quick reset to equal weight (useful after extreme moves)
        catalog.append(DeltaAction(
            name='rebalance_equal_all',
            type='rebalance_equal',
            params={'k': None}  # All assets
        ))
        catalog.append(DeltaAction(
            name='rebalance_equal_top5',
            type='rebalance_equal',
            params={'k': 5}  # Top-5 only
        ))
        catalog.append(DeltaAction(
            name='rebalance_equal_top10',
            type='rebalance_equal',
            params={'k': 10}  # Top-10 only
        ))
        
        # ================================================================
        # 13. SHIFT TO TOP-K (3 actions)
        # ================================================================
        # Move all weight to top-K assets equally
        for k in [3, 5, 7]:
            catalog.append(DeltaAction(
                name=f'shift_to_top{k}',
                type='shift_to_topk',
                params={'k': k}
            ))
        
        # Validate catalog size
        assert len(catalog) == 70, f"Expected 70 actions, got {len(catalog)}"
        
        return catalog
    
    def apply_action(self, action_idx: int, obs: Dict[str, Any], 
                     prev_weights: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Apply delta action to previous portfolio weights.
        
        Parameters
        ----------
        action_idx : int
            Index into catalog (0 to size-1)
        obs : dict
            Environment observation with keys:
            - 'asset_ids': List of tradable assets
            - 'features': [A_t, 4, 60] OHLCV tensor
        prev_weights : np.ndarray
            Previous portfolio weights [A_t]
        
        Returns
        -------
        new_weights : np.ndarray
            Updated portfolio weights [A_t], summing to 1.0
        
        Raises
        ------
        ValueError
            If action_idx is out of bounds
        """
        if not 0 <= action_idx < self.size:
            raise ValueError(f"action_idx {action_idx} out of bounds [0, {self.size})")
        
        action = self.actions[action_idx]
        A_t = len(obs['asset_ids'])
        
        # Validate prev_weights shape
        if len(prev_weights) != A_t:
            raise ValueError(f"prev_weights length {len(prev_weights)} != A_t {A_t}")
        
        # Delegate to action-specific handler
        if action.type == 'hold':
            return self._apply_hold(prev_weights, action.params, A_t)
        elif action.type == 'adjust_topk':
            return self._apply_adjust_topk(prev_weights, action.params, A_t)
        elif action.type == 'rotate':
            return self._apply_rotate(prev_weights, action.params, A_t)
        elif action.type == 'diversify':
            return self._apply_diversify(prev_weights, action.params, A_t)
        elif action.type == 'concentrate':
            return self._apply_concentrate(prev_weights, action.params, A_t)
        elif action.type == 'rebalance_equal':
            return self._apply_rebalance_equal(prev_weights, action.params, A_t)
        elif action.type == 'shift_to_topk':
            return self._apply_shift_to_topk(prev_weights, action.params, A_t)
        else:
            raise ValueError(f"Unknown action type: {action.type}")
    
    # ====================================================================
    # ACTION IMPLEMENTATIONS
    # ====================================================================
    
    def _apply_hold(self, prev_weights: npt.NDArray, params: Dict, 
                    A_t: int) -> npt.NDArray[np.float32]:
        """No change to portfolio."""
        return prev_weights.copy().astype(np.float32)
    
    def _apply_adjust_topk(self, prev_weights: npt.NDArray, params: Dict, 
                           A_t: int) -> npt.NDArray[np.float32]:
        """
        Adjust top K assets by delta_pct.
        
        Example:
            prev_weights = [0.25, 0.20, 0.15, 0.15, 0.10, 0.10, 0.05]
            k=1, delta=+0.10
            → new_weights = [0.35, 0.18, 0.14, 0.14, 0.09, 0.09, 0.05] (renormalized)
        """
        k = min(params['k'], A_t)
        delta = params['delta']
        
        new_weights = prev_weights.copy()
        
        # Add delta to top k assets
        new_weights[:k] += delta / k
        
        # Ensure non-negative
        new_weights = np.maximum(new_weights, 0.0)
        
        # Renormalize
        if new_weights.sum() > 0:
            new_weights /= new_weights.sum()
        else:
            # Fallback to equal weight if all zeros
            new_weights = np.ones(A_t) / A_t
        
        return new_weights.astype(np.float32)
    
    def _apply_rotate(self, prev_weights: npt.NDArray, params: Dict, 
                      A_t: int) -> npt.NDArray[np.float32]:
        """
        Shift weight from one asset to another.
        
        Example:
            prev_weights = [0.30, 0.25, 0.20, 0.15, 0.10]
            from_idx=0, to_idx=1, amount=0.10
            → new_weights = [0.20, 0.35, 0.20, 0.15, 0.10]
        """
        from_idx = min(params['from_idx'], A_t - 1)
        to_idx = min(params['to_idx'], A_t - 1)
        amount = params['amount']
        
        new_weights = prev_weights.copy()
        
        # Shift amount from from_idx to to_idx
        shift = min(amount, new_weights[from_idx])  # Can't shift more than available
        new_weights[from_idx] -= shift
        new_weights[to_idx] += shift
        
        # Ensure non-negative
        new_weights = np.maximum(new_weights, 0.0)
        
        # Renormalize (should already sum to 1, but for safety)
        if new_weights.sum() > 0:
            new_weights /= new_weights.sum()
        
        return new_weights.astype(np.float32)
    
    def _apply_diversify(self, prev_weights: npt.NDArray, params: Dict, 
                         A_t: int) -> npt.NDArray[np.float32]:
        """
        Move weight from concentrated assets toward equal distribution.
        
        Example:
            prev_weights = [0.40, 0.25, 0.15, 0.10, 0.10]  # Concentrated
            amount=0.10
            → Take 0.10 from above-average assets, distribute equally
            → new_weights ≈ [0.32, 0.21, 0.17, 0.15, 0.15]  # More equal
        """
        amount = params['amount']
        
        # Target: equal weight
        equal_weight = 1.0 / A_t
        
        # Compute gap (assets above equal weight)
        above_equal = prev_weights > equal_weight
        
        if not above_equal.any():
            # Already equal or more diversified, no change
            return prev_weights.copy().astype(np.float32)
        
        # Take 'amount' from above-equal assets proportionally
        excess = prev_weights - equal_weight
        excess[~above_equal] = 0.0
        
        # Scale excess to total 'amount'
        if excess.sum() > 0:
            scaling = min(amount / excess.sum(), 1.0)
            redistribution = excess * scaling
        else:
            return prev_weights.copy().astype(np.float32)
        
        # New weights: reduce above-equal, increase below-equal
        new_weights = prev_weights - redistribution
        new_weights += redistribution.sum() / A_t  # Distribute equally
        
        # Ensure non-negative
        new_weights = np.maximum(new_weights, 0.0)
        
        # Renormalize
        if new_weights.sum() > 0:
            new_weights /= new_weights.sum()
        
        return new_weights.astype(np.float32)
    
    def _apply_concentrate(self, prev_weights: npt.NDArray, params: Dict, 
                           A_t: int) -> npt.NDArray[np.float32]:
        """
        Move weight from diversified assets toward top asset.
        
        Example:
            prev_weights = [0.20, 0.20, 0.20, 0.20, 0.20]  # Equal
            amount=0.10
            → Take 0.10 from below-average assets, give to top-1
            → new_weights ≈ [0.30, 0.175, 0.175, 0.175, 0.175]  # Concentrated
        """
        amount = params['amount']
        
        # Take 'amount' from bottom assets, give to top asset
        equal_weight = 1.0 / A_t
        below_equal = prev_weights < equal_weight
        
        if not below_equal.any():
            # Already concentrated, no change
            return prev_weights.copy().astype(np.float32)
        
        # Take proportionally from below-equal assets
        deficit = equal_weight - prev_weights
        deficit[~below_equal] = 0.0
        
        if deficit.sum() > 0:
            scaling = min(amount / deficit.sum(), 1.0)
            take_from = deficit * scaling
        else:
            return prev_weights.copy().astype(np.float32)
        
        # New weights: reduce below-equal, increase top-1
        new_weights = prev_weights - take_from
        new_weights[0] += take_from.sum()
        
        # Ensure non-negative
        new_weights = np.maximum(new_weights, 0.0)
        
        # Renormalize
        if new_weights.sum() > 0:
            new_weights /= new_weights.sum()
        
        return new_weights.astype(np.float32)
    
    def _apply_rebalance_equal(self, prev_weights: npt.NDArray, params: Dict, 
                               A_t: int) -> npt.NDArray[np.float32]:
        """
        Reset to equal weight among top-K or all assets.
        
        Example:
            prev_weights = [0.40, 0.25, 0.15, 0.10, 0.10]
            k=None (all assets)
            → new_weights = [0.20, 0.20, 0.20, 0.20, 0.20]
        """
        k = params['k'] if params['k'] else A_t
        k = min(k, A_t)
        
        new_weights = np.zeros(A_t, dtype=np.float32)
        new_weights[:k] = 1.0 / k
        return new_weights
    
    def _apply_shift_to_topk(self, prev_weights: npt.NDArray, params: Dict, 
                             A_t: int) -> npt.NDArray[np.float32]:
        """
        Zero out bottom assets, equal weight top-K.
        
        Example:
            prev_weights = [0.25, 0.20, 0.15, 0.15, 0.10, 0.10, 0.05]
            k=3
            → new_weights = [0.333, 0.333, 0.333, 0, 0, 0, 0]
        """
        k = min(params['k'], A_t)
        
        new_weights = np.zeros(A_t, dtype=np.float32)
        new_weights[:k] = 1.0 / k
        return new_weights
    
    def get_action_name(self, action_idx: int) -> str:
        """
        Get human-readable name for a catalog action.
        
        Parameters
        ----------
        action_idx : int
            Action index
        
        Returns
        -------
        name : str
            Action name
        """
        if not 0 <= action_idx < self.size:
            raise ValueError(f"action_idx {action_idx} out of bounds")
        return self.actions[action_idx].name
    
    def __repr__(self) -> str:
        return f"DeltaActionCatalog(size={self.size}, actions={self.size})"
