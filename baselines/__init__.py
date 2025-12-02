"""
Baseline portfolio strategies for comparison with RL agents.

This module provides classical portfolio optimization baselines that operate
under the same constraints as RL agents (transaction costs, turnover limits,
concentration caps) for fair comparison.

Available Baselines:
- EqualWeightAgent: Uniform 1/N allocation across tradable assets
- MarketCapWeightAgent: Market-cap weighted allocation using index weights
- MeanVarianceAgent: Markowitz mean-variance optimization with constraints

All baselines implement the BaseAgent interface and can be evaluated using
the same environment and metrics infrastructure as RL agents.
"""

from baselines.equal_weight import EqualWeightAgent
from baselines.market_cap_weight import MarketCapWeightAgent
from baselines.mean_variance import MeanVarianceAgent

__all__ = [
    "EqualWeightAgent",
    "MarketCapWeightAgent",
    "MeanVarianceAgent",
]
