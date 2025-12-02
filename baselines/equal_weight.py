"""
Equal Weight (1/N) Baseline Portfolio Strategy.

This baseline allocates equal weight to all tradable assets at each rebalancing
period. It's one of the most robust baselines in portfolio optimization and
often outperforms more sophisticated strategies [DeMiguel et al., 2009].

The strategy maintains universe constraints by:
- Only allocating to assets in the current tradable universe
- Respecting turnover caps via environment projection
- Paying transaction costs like all other agents

References:
- DeMiguel, V., Garlappi, L., & Uppal, R. (2009). "Optimal Versus Naive
  Diversification: How Inefficient is the 1/N Portfolio Strategy?"
  Review of Financial Studies.
"""

import numpy as np
import numpy.typing as npt

from baselines.base_baseline import BaselineAgent, BaselineConfig
from environment.environment import Obs


class EqualWeightConfig(BaselineConfig):
    """Configuration for Equal Weight baseline."""
    
    def __init__(
        self,
        random_seed: int = 42,
        log_dir=None,
    ):
        super().__init__(
            name="EqualWeight",
            random_seed=random_seed,
            log_dir=log_dir,
        )


class EqualWeightAgent(BaselineAgent):
    """
    Equal-weight (1/N) portfolio baseline.
    
    At each decision point, allocates equal weight to all A_t tradable assets:
        w_i = 1 / A_t  for all i in tradable universe
    
    This is a strong baseline because:
    1. No estimation error (no parameters to estimate)
    2. Maximum diversification within the tradable universe
    3. Automatic rebalancing to maintain equal weights
    
    The environment handles constraint enforcement (turnover caps, etc.)
    via projection, so this agent simply proposes equal weights and
    the environment adjusts as needed.
    
    Example
    -------
    >>> from baselines import EqualWeightAgent
    >>> from baselines.base_baseline import BaselineConfig
    >>> 
    >>> config = EqualWeightConfig(random_seed=42)
    >>> agent = EqualWeightAgent(config)
    >>> 
    >>> # Run evaluation
    >>> results = agent.evaluate(env, n_episodes=1)
    >>> print(f"Return: {results['mean_return']:.4f}")
    """
    
    def __init__(self, config: EqualWeightConfig = None):
        """
        Initialize Equal Weight agent.
        
        Parameters
        ----------
        config : EqualWeightConfig, optional
            Agent configuration. If None, uses defaults.
        """
        if config is None:
            config = EqualWeightConfig()
        super().__init__(config)
    
    def select_action(self, obs: Obs) -> npt.NDArray[np.float32]:
        """
        Select equal-weight allocation.
        
        Parameters
        ----------
        obs : Obs
            Current observation with asset_ids
            
        Returns
        -------
        weights : np.ndarray
            Equal weights [1/A_t, 1/A_t, ..., 1/A_t]
        """
        n_assets = len(obs['asset_ids'])
        
        if n_assets == 0:
            return np.array([], dtype=np.float32)
        
        weights = np.full(n_assets, 1.0 / n_assets, dtype=np.float32)
        return weights
    
    def get_name(self) -> str:
        """Return strategy name."""
        return "Equal Weight (1/N)"
