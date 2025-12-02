"""
Mean-Variance Optimization Baseline Portfolio Strategy.

This baseline implements Markowitz mean-variance optimization using
historical returns estimated from the observation tensor. It finds
weights that maximize expected return for a given level of risk
(or minimize risk for a given return target).

Optimization Problem (Minimum Variance with Return Target):
    minimize    w^T Σ w
    subject to  w^T μ >= target_return
                w >= 0  (long-only)
                Σ w_i = 1  (fully invested)
                w_i <= max_weight  (concentration limit)

The strategy uses:
- Rolling historical returns from the 60-day observation window
- Shrinkage covariance estimation for stability
- CVXPY for constrained quadratic optimization
- Fallback to equal weight when optimization fails

References:
- Markowitz, H. (1952). "Portfolio Selection." The Journal of Finance.
- Ledoit, O., & Wolf, M. (2004). "A well-conditioned estimator for
  large-dimensional covariance matrices." Journal of Multivariate Analysis.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
from pathlib import Path

import numpy as np
import numpy.typing as npt

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    cp = None

from baselines.base_baseline import BaselineAgent, BaselineConfig
from environment.environment import Obs


@dataclass 
class MeanVarianceConfig(BaselineConfig):
    """
    Configuration for Mean-Variance Optimization baseline.
    
    Attributes
    ----------
    risk_aversion : float
        Risk aversion parameter γ. Higher values prefer lower variance.
        - γ = 0: Maximize return only (aggressive)
        - γ = 1: Balance return and variance
        - γ = 10: Strongly prefer low variance (conservative)
    target_return : float, optional
        If set, solve minimum variance subject to return >= target.
        If None, solve mean-variance with risk_aversion parameter.
    shrinkage_factor : float
        Ledoit-Wolf style shrinkage toward diagonal matrix.
        0.0 = no shrinkage (use sample covariance)
        1.0 = full shrinkage (use diagonal)
        Recommended: 0.2-0.5 for stability
    max_weight_per_asset : float
        Maximum weight per asset (concentration limit).
    lookback_window : int
        Number of days to use for return estimation.
        Should match the observation window (60 days).
    min_history : int
        Minimum number of valid returns required for optimization.
        Falls back to equal weight if insufficient data.
    use_log_returns : bool
        If True, estimate returns using log returns.
        If False, use simple returns.
    """
    risk_aversion: float = 1.0
    target_return: Optional[float] = None
    shrinkage_factor: float = 0.3
    max_weight_per_asset: float = 0.35
    lookback_window: int = 60
    min_history: int = 20
    use_log_returns: bool = True
    
    def __init__(
        self,
        random_seed: int = 42,
        log_dir: Optional[Path] = None,
        risk_aversion: float = 1.0,
        target_return: Optional[float] = None,
        shrinkage_factor: float = 0.3,
        max_weight_per_asset: float = 0.35,
        lookback_window: int = 60,
        min_history: int = 20,
        use_log_returns: bool = True,
    ):
        super().__init__(
            name="MeanVariance",
            random_seed=random_seed,
            log_dir=log_dir,
        )
        self.risk_aversion = risk_aversion
        self.target_return = target_return
        self.shrinkage_factor = shrinkage_factor
        self.max_weight_per_asset = max_weight_per_asset
        self.lookback_window = lookback_window
        self.min_history = min_history
        self.use_log_returns = use_log_returns


class MeanVarianceAgent(BaselineAgent):
    """
    Mean-Variance Optimization portfolio baseline.
    
    Computes optimal portfolio weights using Markowitz mean-variance
    optimization with the following constraints:
    - Long-only (no short selling)
    - Fully invested (weights sum to 1)
    - Concentration limit (per-asset cap)
    
    Returns are estimated from the 60-day observation window, and
    covariance is shrunk toward a diagonal matrix for stability.
    
    The optimization uses CVXPY, which must be installed:
        pip install cvxpy
    
    Example
    -------
    >>> from baselines import MeanVarianceAgent
    >>> from baselines.mean_variance import MeanVarianceConfig
    >>> 
    >>> config = MeanVarianceConfig(
    ...     risk_aversion=2.0,
    ...     shrinkage_factor=0.3,
    ...     max_weight_per_asset=0.35,
    ... )
    >>> agent = MeanVarianceAgent(config)
    >>> 
    >>> # Run evaluation
    >>> results = agent.evaluate(env, n_episodes=1)
    >>> print(f"Return: {results['mean_return']:.4f}")
    """
    
    def __init__(self, config: MeanVarianceConfig = None):
        """
        Initialize Mean-Variance agent.
        
        Parameters
        ----------
        config : MeanVarianceConfig, optional
            Agent configuration. If None, uses defaults.
            
        Raises
        ------
        ImportError
            If cvxpy is not installed.
        """
        if not CVXPY_AVAILABLE:
            raise ImportError(
                "cvxpy is required for MeanVarianceAgent. "
                "Install with: pip install cvxpy"
            )
        
        if config is None:
            config = MeanVarianceConfig()
        super().__init__(config)
        
        self.config: MeanVarianceConfig = config
        
        # Track optimization statistics
        self._n_optimizations = 0
        self._n_fallbacks = 0
    
    def _extract_returns(
        self, 
        features: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """
        Extract daily returns from observation tensor.
        
        The observation tensor has shape [A_t, 4, 60] with channels:
        - 0: Close (normalized so last day = 1.0)
        - 1: High (normalized)
        - 2: Low (normalized)
        - 3: Volume (z-scored log volume)
        
        Since prices are normalized by the last close, we can compute
        returns from the close channel.
        
        Parameters
        ----------
        features : np.ndarray
            Observation tensor [A_t, 4, 60]
            
        Returns
        -------
        returns : np.ndarray
            Daily returns [A_t, T-1] where T is lookback window
        """
        # Extract close prices (channel 0)
        # Shape: [A_t, 60]
        close_prices = features[:, 0, :]
        
        # Compute returns
        if self.config.use_log_returns:
            # Log returns: r_t = log(p_t / p_{t-1})
            # Handle potential zeros/negatives
            close_safe = np.maximum(close_prices, 1e-8)
            returns = np.diff(np.log(close_safe), axis=1)
        else:
            # Simple returns: r_t = (p_t - p_{t-1}) / p_{t-1}
            close_safe = np.maximum(close_prices[:, :-1], 1e-8)
            returns = np.diff(close_prices, axis=1) / close_safe
        
        return returns.astype(np.float32)
    
    def _estimate_moments(
        self, 
        returns: npt.NDArray[np.float32]
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Estimate expected returns and covariance matrix with shrinkage.
        
        Parameters
        ----------
        returns : np.ndarray
            Daily returns [A_t, T-1]
            
        Returns
        -------
        mu : np.ndarray
            Expected returns [A_t]
        sigma : np.ndarray
            Covariance matrix [A_t, A_t]
        """
        n_assets, n_obs = returns.shape
        
        # Expected returns (sample mean)
        mu = np.mean(returns, axis=1).astype(np.float64)
        
        # Sample covariance
        returns_centered = returns - mu[:, np.newaxis]
        sample_cov = (returns_centered @ returns_centered.T) / max(n_obs - 1, 1)
        
        # Shrinkage toward diagonal (Ledoit-Wolf style)
        # Target: diagonal matrix with average variance
        diag_target = np.diag(np.diag(sample_cov))
        
        shrinkage = self.config.shrinkage_factor
        sigma = (1 - shrinkage) * sample_cov + shrinkage * diag_target
        
        # Ensure positive semi-definite
        sigma = sigma.astype(np.float64)
        min_eig = np.min(np.linalg.eigvalsh(sigma))
        if min_eig < 1e-8:
            sigma += (1e-8 - min_eig) * np.eye(n_assets)
        
        return mu, sigma
    
    def _solve_mean_variance(
        self,
        mu: npt.NDArray[np.float64],
        sigma: npt.NDArray[np.float64],
        n_assets: int,
    ) -> Optional[npt.NDArray[np.float32]]:
        """
        Solve mean-variance optimization using CVXPY.
        
        Two formulations supported:
        1. If target_return is set: Minimum variance s.t. return >= target
        2. Otherwise: Maximize (μ^T w - γ/2 * w^T Σ w)
        
        Parameters
        ----------
        mu : np.ndarray
            Expected returns [A_t]
        sigma : np.ndarray
            Covariance matrix [A_t, A_t]
        n_assets : int
            Number of assets
            
        Returns
        -------
        weights : np.ndarray or None
            Optimal weights [A_t], or None if optimization failed
        """
        # Decision variable
        w = cp.Variable(n_assets)
        
        # Constraints
        constraints = [
            w >= 0,  # Long-only
            cp.sum(w) == 1,  # Fully invested
            w <= self.config.max_weight_per_asset,  # Concentration limit
        ]
        
        # Objective
        if self.config.target_return is not None:
            # Minimum variance with return target
            constraints.append(mu @ w >= self.config.target_return)
            objective = cp.Minimize(cp.quad_form(w, sigma))
        else:
            # Mean-variance utility: maximize return - risk_aversion * variance
            gamma = self.config.risk_aversion
            # Note: CVXPY uses Minimize, so we negate the utility
            portfolio_return = mu @ w
            portfolio_variance = cp.quad_form(w, sigma)
            objective = cp.Maximize(portfolio_return - (gamma / 2) * portfolio_variance)
        
        # Solve
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=cp.OSQP, verbose=False)
            
            if problem.status in ['optimal', 'optimal_inaccurate']:
                weights = w.value
                if weights is not None:
                    # Clip small negatives and renormalize
                    weights = np.maximum(weights, 0)
                    weights = weights / weights.sum()
                    return weights.astype(np.float32)
        except Exception as e:
            # Optimization failed
            pass
        
        return None
    
    def _solve_minimum_variance(
        self,
        sigma: npt.NDArray[np.float64],
        n_assets: int,
    ) -> Optional[npt.NDArray[np.float32]]:
        """
        Solve minimum variance portfolio (no return target).
        
        This is a fallback when mean-variance fails or when
        return estimates are unreliable.
        
        Parameters
        ----------
        sigma : np.ndarray
            Covariance matrix [A_t, A_t]
        n_assets : int
            Number of assets
            
        Returns
        -------
        weights : np.ndarray or None
            Minimum variance weights, or None if failed
        """
        w = cp.Variable(n_assets)
        
        constraints = [
            w >= 0,
            cp.sum(w) == 1,
            w <= self.config.max_weight_per_asset,
        ]
        
        objective = cp.Minimize(cp.quad_form(w, sigma))
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=cp.OSQP, verbose=False)
            
            if problem.status in ['optimal', 'optimal_inaccurate']:
                weights = w.value
                if weights is not None:
                    weights = np.maximum(weights, 0)
                    weights = weights / weights.sum()
                    return weights.astype(np.float32)
        except Exception:
            pass
        
        return None
    
    def select_action(self, obs: Obs) -> npt.NDArray[np.float32]:
        """
        Select mean-variance optimized allocation.
        
        Parameters
        ----------
        obs : Obs
            Current observation with features tensor
            
        Returns
        -------
        weights : np.ndarray
            Optimized portfolio weights
        """
        features = obs['features']
        n_assets = len(obs['asset_ids'])
        
        if n_assets == 0:
            return np.array([], dtype=np.float32)
        
        # Equal weight fallback
        equal_weights = np.full(n_assets, 1.0 / n_assets, dtype=np.float32)
        
        # Extract returns from observation tensor
        returns = self._extract_returns(features)
        
        # Check for sufficient data
        n_obs = returns.shape[1]
        if n_obs < self.config.min_history:
            self._n_fallbacks += 1
            return equal_weights
        
        # Check for valid returns (no NaN/Inf)
        if not np.all(np.isfinite(returns)):
            self._n_fallbacks += 1
            return equal_weights
        
        # Estimate moments
        try:
            mu, sigma = self._estimate_moments(returns)
        except Exception:
            self._n_fallbacks += 1
            return equal_weights
        
        # Solve optimization
        weights = self._solve_mean_variance(mu, sigma, n_assets)
        
        if weights is None:
            # Try minimum variance as fallback
            weights = self._solve_minimum_variance(sigma, n_assets)
        
        if weights is None:
            # Final fallback to equal weight
            self._n_fallbacks += 1
            return equal_weights
        
        self._n_optimizations += 1
        return weights
    
    def get_name(self) -> str:
        """Return strategy name."""
        if self.config.target_return is not None:
            return f"Mean-Variance (target={self.config.target_return:.4f})"
        return f"Mean-Variance (γ={self.config.risk_aversion})"
    
    def get_optimization_stats(self) -> dict:
        """
        Get optimization statistics.
        
        Returns
        -------
        stats : dict
            Dictionary with optimization and fallback counts
        """
        total = self._n_optimizations + self._n_fallbacks
        return {
            'n_optimizations': self._n_optimizations,
            'n_fallbacks': self._n_fallbacks,
            'optimization_rate': self._n_optimizations / max(total, 1),
        }
