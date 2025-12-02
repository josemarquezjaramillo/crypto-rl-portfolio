#!/usr/bin/env python
"""
Evaluator module for comprehensive agent evaluation.

Runs all agents (baselines + RL) across multiple seeds and validation windows,
computing metrics with confidence intervals in the style of Lucarelli (2020)
and Jiang (2017).

Usage:
    python -m evaluation.evaluator --split val --output results/
    python -m evaluation.evaluator --split test --output results/
"""
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
import numpy as np
import pandas as pd
from datetime import datetime
import json

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import metrics module
from evaluation.metrics import (
    compute_all_metrics,
    compute_cagr,
    compute_annualized_volatility,
    compute_sortino_ratio,
    compute_calmar_ratio,
    compute_hit_rate,
    compute_confidence_interval,
)

from data.dataset_loader import load_exported_dataset
from data.dataset_backend import DatasetBackend
from environment.environment import PortfolioEnv, EnvConfig

# Baselines
from baselines.equal_weight import EqualWeightAgent, EqualWeightConfig
from baselines.market_cap_weight import MarketCapWeightAgent, MarketCapWeightConfig
from baselines.mean_variance import MeanVarianceAgent, MeanVarianceConfig

# RL Agents
from agents.dqn.dqn_agent import DQNAgent, DQNConfig


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class EvaluationConfig:
    """Configuration for evaluation runs."""
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1024])
    cost_rate: float = 0.001
    turnover_cap: float = 0.30
    max_weight_per_asset: float = 0.35
    dataset_path: str = "dataset_v1"
    output_dir: Path = field(default_factory=lambda: Path("results"))
    device: str = "cuda"
    
    def __post_init__(self):
        self.output_dir = Path(self.output_dir)


@dataclass
class AgentResult:
    """Results from a single agent run (one seed, one window)."""
    agent_name: str
    agent_type: str  # "baseline" or "rl"
    window_name: str  # "val_2018_crash", "combined", etc.
    seed: int
    n_steps: int
    
    # Profitability metrics
    cumulative_return: float
    cagr: float  # Compound Annual Growth Rate
    
    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    annualized_volatility: float
    
    # Efficiency metrics
    hit_rate: float  # Fraction of profitable days
    mean_turnover: float
    total_costs: float
    
    # Time series (for visualization)
    portfolio_values: List[float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'agent_name': self.agent_name,
            'agent_type': self.agent_type,
            'window_name': self.window_name,
            'seed': self.seed,
            'n_steps': self.n_steps,
            'cumulative_return': self.cumulative_return,
            'cagr': self.cagr,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'calmar_ratio': self.calmar_ratio,
            'annualized_volatility': self.annualized_volatility,
            'hit_rate': self.hit_rate,
            'mean_turnover': self.mean_turnover,
            'total_costs': self.total_costs,
        }


@dataclass
class DetailedAgentResult:
    """
    Detailed results from a single agent run with full time-series data.
    
    Used for generating publication-quality visualizations including:
    - Portfolio value evolution charts
    - Asset allocation stacked area charts
    - Drawdown evolution
    - Rolling performance metrics
    
    This extends AgentResult with per-step data needed for visualization.
    """
    # Core identification
    agent_name: str
    agent_type: str  # "baseline" or "rl"
    window_name: str
    seed: int
    n_steps: int
    
    # Summary metrics (same as AgentResult)
    cumulative_return: float
    cagr: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    annualized_volatility: float
    hit_rate: float
    mean_turnover: float
    total_costs: float
    
    # Time series data (per-step)
    dates: List[np.datetime64]  # Trading dates
    portfolio_values: List[float]  # Portfolio value at each step
    daily_returns: List[float]  # Daily log returns
    turnovers_per_step: List[float]  # Turnover at each step
    costs_per_step: List[float]  # Transaction costs per step
    rewards_per_step: List[float]  # Reward at each step
    
    # Weight history for allocation charts
    weights_history: List[Dict[str, float]]  # {asset_id: weight} at each step
    asset_ids_history: List[List[str]]  # Available assets at each step
    
    def to_agent_result(self) -> AgentResult:
        """Convert to simplified AgentResult (drops time-series detail)."""
        return AgentResult(
            agent_name=self.agent_name,
            agent_type=self.agent_type,
            window_name=self.window_name,
            seed=self.seed,
            n_steps=self.n_steps,
            cumulative_return=self.cumulative_return,
            cagr=self.cagr,
            sharpe_ratio=self.sharpe_ratio,
            sortino_ratio=self.sortino_ratio,
            max_drawdown=self.max_drawdown,
            calmar_ratio=self.calmar_ratio,
            annualized_volatility=self.annualized_volatility,
            hit_rate=self.hit_rate,
            mean_turnover=self.mean_turnover,
            total_costs=self.total_costs,
            portfolio_values=self.portfolio_values,
        )
    
    def to_strategy_timeseries(self):
        """Convert to StrategyTimeSeries for visualization functions."""
        from evaluation.visualizer import StrategyTimeSeries
        return StrategyTimeSeries(
            name=self.agent_name,
            dates=self.dates,
            portfolio_values=self.portfolio_values,
            weights_history=self.weights_history,
            asset_ids_history=self.asset_ids_history,
            turnovers=self.turnovers_per_step,
            rewards=self.rewards_per_step,
        )


@dataclass
class AggregatedResult:
    """Aggregated results across multiple seeds."""
    agent_name: str
    agent_type: str
    window_name: str
    n_runs: int
    
    # Profitability metrics
    mean_return: float
    std_return: float
    ci_95_return: Tuple[float, float]
    mean_cagr: float
    std_cagr: float
    
    # Sharpe metrics
    mean_sharpe: float
    std_sharpe: float
    ci_95_sharpe: Tuple[float, float]
    
    # Sortino metrics
    mean_sortino: float
    std_sortino: float
    
    # Risk metrics
    mean_max_dd: float
    std_max_dd: float
    mean_calmar: float
    std_calmar: float
    mean_volatility: float
    std_volatility: float
    
    # Efficiency metrics
    mean_hit_rate: float
    std_hit_rate: float
    mean_turnover: float
    mean_costs: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'agent_name': self.agent_name,
            'agent_type': self.agent_type,
            'window_name': self.window_name,
            'n_runs': self.n_runs,
            'mean_return': self.mean_return,
            'std_return': self.std_return,
            'ci_95_return_lower': self.ci_95_return[0],
            'ci_95_return_upper': self.ci_95_return[1],
            'mean_cagr': self.mean_cagr,
            'std_cagr': self.std_cagr,
            'mean_sharpe': self.mean_sharpe,
            'std_sharpe': self.std_sharpe,
            'ci_95_sharpe_lower': self.ci_95_sharpe[0],
            'ci_95_sharpe_upper': self.ci_95_sharpe[1],
            'mean_sortino': self.mean_sortino,
            'std_sortino': self.std_sortino,
            'mean_max_dd': self.mean_max_dd,
            'std_max_dd': self.std_max_dd,
            'mean_calmar': self.mean_calmar,
            'std_calmar': self.std_calmar,
            'mean_volatility': self.mean_volatility,
            'std_volatility': self.std_volatility,
            'mean_hit_rate': self.mean_hit_rate,
            'std_hit_rate': self.std_hit_rate,
            'mean_turnover': self.mean_turnover,
            'mean_costs': self.mean_costs,
        }


# =============================================================================
# Agent Registry
# =============================================================================

# Type alias for agent factory functions
AgentFactory = Callable[[int, PortfolioEnv], Any]


def create_equal_weight_agent(seed: int, env: PortfolioEnv) -> EqualWeightAgent:
    """Factory for Equal Weight baseline."""
    config = EqualWeightConfig(random_seed=seed)
    return EqualWeightAgent(config)


def create_market_cap_agent(seed: int, env: PortfolioEnv, max_weight: float = 0.35) -> MarketCapWeightAgent:
    """Factory for Market Cap Weight baseline."""
    config = MarketCapWeightConfig(random_seed=seed, max_weight_per_asset=max_weight)
    return MarketCapWeightAgent(config)


def create_mean_variance_agent(seed: int, env: PortfolioEnv, 
                                risk_aversion: float = 1.0,
                                max_weight: float = 0.35) -> MeanVarianceAgent:
    """Factory for Mean-Variance baseline."""
    config = MeanVarianceConfig(
        random_seed=seed, 
        risk_aversion=risk_aversion,
        max_weight_per_asset=max_weight
    )
    return MeanVarianceAgent(config)


def create_dqn_agent(seed: int, env: PortfolioEnv, 
                     checkpoint_path: Path,
                     device: str = "cuda",
                     use_double: bool = False) -> DQNAgent:
    """Factory for DQN/DDQN agent."""
    config = DQNConfig(
        name="DQN" if not use_double else "DDQN",
        random_seed=seed,
        device=device,
        dataset_path="dataset_v1",
        hidden_dims=[256, 128],  # Match checkpoint architecture
        use_double_dqn=use_double,
    )
    agent = DQNAgent(config, env)
    agent.load(checkpoint_path)
    agent.epsilon = 0.0  # Deterministic evaluation
    return agent


# =============================================================================
# Core Evaluation Logic
# =============================================================================

def compute_confidence_interval(values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """
    Compute confidence interval for a list of values.
    
    Uses t-distribution for small samples (n < 30).
    """
    n = len(values)
    if n < 2:
        mean_val = values[0] if values else 0.0
        return (mean_val, mean_val)
    
    mean_val = np.mean(values)
    std_val = np.std(values, ddof=1)  # Sample std
    
    # For 95% CI with small samples, use t-distribution
    # t-value for 95% CI: ~2.0 for n=5, ~1.96 for large n
    from scipy import stats
    t_value = stats.t.ppf((1 + confidence) / 2, df=n-1)
    
    margin = t_value * std_val / np.sqrt(n)
    return (mean_val - margin, mean_val + margin)


def run_single_evaluation(
    agent,
    env: PortfolioEnv,
    seed: int,
    agent_name: str,
    agent_type: str,
    window_name: str,
) -> AgentResult:
    """
    Run a single evaluation episode.
    
    Parameters
    ----------
    agent : BaselineAgent or DQNAgent
        Agent to evaluate (must have select_action method)
    env : PortfolioEnv
        Environment instance
    seed : int
        Random seed for this run
    agent_name : str
        Human-readable agent name
    agent_type : str
        "baseline" or "rl"
    window_name : str
        Validation window identifier
        
    Returns
    -------
    AgentResult
        Evaluation metrics for this run
    """
    obs = env.reset(seed=seed)
    done = False
    
    rewards = []
    pvs = [1.0]  # Start at 1.0
    turnovers = []
    costs = []
    
    while not done:
        # Get action based on agent type
        if hasattr(agent, 'select_action'):
            # DQN-style agent
            if agent_type == "rl":
                action = agent.select_action(obs, deterministic=True)
            else:
                action = agent.select_action(obs)
        else:
            raise ValueError(f"Agent {agent_name} has no select_action method")
        
        next_obs, reward, done, info = env.step(action)
        
        rewards.append(reward)
        pvs.append(info.get('portfolio_value', pvs[-1]))
        turnovers.append(info.get('turnover', 0.0))
        costs.append(info.get('transaction_cost', 0.0))
        
        obs = next_obs
    
    # Compute all metrics using metrics module
    pvs_arr = np.array(pvs)
    metrics = compute_all_metrics(pvs_arr, turnovers, costs)
    
    return AgentResult(
        agent_name=agent_name,
        agent_type=agent_type,
        window_name=window_name,
        seed=seed,
        n_steps=metrics['n_steps'],
        cumulative_return=metrics['cumulative_return'],
        cagr=metrics['cagr'],
        sharpe_ratio=metrics['sharpe_ratio'],
        sortino_ratio=metrics['sortino_ratio'],
        max_drawdown=metrics['max_drawdown'],
        calmar_ratio=metrics['calmar_ratio'],
        annualized_volatility=metrics['annualized_volatility'],
        hit_rate=metrics['hit_rate'],
        mean_turnover=metrics['mean_turnover'],
        total_costs=metrics['total_costs'],
        portfolio_values=pvs,
    )


def run_detailed_evaluation(
    agent,
    env: PortfolioEnv,
    seed: int,
    agent_name: str,
    agent_type: str,
    window_name: str,
) -> DetailedAgentResult:
    """
    Run a single evaluation episode with full time-series data collection.
    
    This function collects detailed per-step data needed for publication-quality
    visualizations including allocation charts, drawdown evolution, and rolling
    performance metrics.
    
    Parameters
    ----------
    agent : BaselineAgent or DQNAgent
        Agent to evaluate (must have select_action method)
    env : PortfolioEnv
        Environment instance
    seed : int
        Random seed for this run
    agent_name : str
        Human-readable agent name
    agent_type : str
        "baseline" or "rl"
    window_name : str
        Validation window identifier
        
    Returns
    -------
    DetailedAgentResult
        Evaluation metrics with full time-series data
    """
    obs = env.reset(seed=seed)
    done = False
    
    # Time series collectors
    dates = [obs['date']]
    rewards = []
    pvs = [1.0]  # Start at 1.0
    turnovers = []
    costs = []
    
    # Weight history for allocation charts
    # Initial weights from observation
    weights_history = []
    asset_ids_history = [obs['asset_ids'].copy()]
    
    # Store initial weights (prev_weights from first obs)
    initial_weights = {
        asset_id: float(w) 
        for asset_id, w in zip(obs['asset_ids'], obs['prev_weights'])
    }
    weights_history.append(initial_weights)
    
    while not done:
        # Get action based on agent type
        if hasattr(agent, 'select_action'):
            if agent_type == "rl":
                action = agent.select_action(obs, deterministic=True)
            else:
                action = agent.select_action(obs)
        else:
            raise ValueError(f"Agent {agent_name} has no select_action method")
        
        next_obs, reward, done, info = env.step(action)
        
        # Collect step data
        rewards.append(reward)
        pvs.append(info.get('portfolio_value', pvs[-1]))
        turnovers.append(info.get('turnover', 0.0))
        costs.append(info.get('transaction_cost', 0.0))
        
        # Collect date
        if not done:
            dates.append(next_obs['date'])
        else:
            # For final step, use info date if available
            dates.append(info.get('date', dates[-1]))
        
        # Collect executed weights
        executed_weights = info.get('executed_weights', None)
        asset_ids = info.get('tradable_assets', obs.get('asset_ids', []))
        
        if executed_weights is not None and len(asset_ids) > 0:
            weights_dict = {
                asset_id: float(w)
                for asset_id, w in zip(asset_ids, executed_weights)
            }
            weights_history.append(weights_dict)
            asset_ids_history.append(list(asset_ids))
        else:
            # Fallback: copy previous weights
            weights_history.append(weights_history[-1].copy())
            asset_ids_history.append(asset_ids_history[-1].copy())
        
        obs = next_obs
    
    # Compute daily returns
    pvs_arr = np.array(pvs)
    daily_returns = list(np.diff(np.log(pvs_arr + 1e-10)))  # Log returns
    
    # Compute all metrics
    metrics = compute_all_metrics(pvs_arr, turnovers, costs)
    
    return DetailedAgentResult(
        agent_name=agent_name,
        agent_type=agent_type,
        window_name=window_name,
        seed=seed,
        n_steps=metrics['n_steps'],
        cumulative_return=metrics['cumulative_return'],
        cagr=metrics['cagr'],
        sharpe_ratio=metrics['sharpe_ratio'],
        sortino_ratio=metrics['sortino_ratio'],
        max_drawdown=metrics['max_drawdown'],
        calmar_ratio=metrics['calmar_ratio'],
        annualized_volatility=metrics['annualized_volatility'],
        hit_rate=metrics['hit_rate'],
        mean_turnover=metrics['mean_turnover'],
        total_costs=metrics['total_costs'],
        dates=dates,
        portfolio_values=pvs,
        daily_returns=daily_returns,
        turnovers_per_step=turnovers,
        costs_per_step=costs,
        rewards_per_step=rewards,
        weights_history=weights_history,
        asset_ids_history=asset_ids_history,
    )


def aggregate_results(results: List[AgentResult]) -> AggregatedResult:
    """
    Aggregate results from multiple runs into summary statistics.
    
    Parameters
    ----------
    results : List[AgentResult]
        Results from multiple seeds (same agent, same window)
        
    Returns
    -------
    AggregatedResult
        Aggregated metrics with confidence intervals
    """
    if not results:
        raise ValueError("No results to aggregate")
    
    # All results should be from same agent/window
    agent_name = results[0].agent_name
    agent_type = results[0].agent_type
    window_name = results[0].window_name
    
    # Extract all metrics
    returns = [r.cumulative_return for r in results]
    cagrs = [r.cagr for r in results]
    sharpes = [r.sharpe_ratio for r in results]
    sortinos = [r.sortino_ratio for r in results]
    max_dds = [r.max_drawdown for r in results]
    calmars = [r.calmar_ratio for r in results]
    volatilities = [r.annualized_volatility for r in results]
    hit_rates = [r.hit_rate for r in results]
    turnovers = [r.mean_turnover for r in results]
    costs = [r.total_costs for r in results]
    
    return AggregatedResult(
        agent_name=agent_name,
        agent_type=agent_type,
        window_name=window_name,
        n_runs=len(results),
        # Profitability
        mean_return=np.mean(returns),
        std_return=np.std(returns),
        ci_95_return=compute_confidence_interval(returns),
        mean_cagr=np.mean(cagrs),
        std_cagr=np.std(cagrs),
        # Sharpe
        mean_sharpe=np.mean(sharpes),
        std_sharpe=np.std(sharpes),
        ci_95_sharpe=compute_confidence_interval(sharpes),
        # Sortino
        mean_sortino=np.mean(sortinos),
        std_sortino=np.std(sortinos),
        # Risk
        mean_max_dd=np.mean(max_dds),
        std_max_dd=np.std(max_dds),
        mean_calmar=np.mean(calmars),
        std_calmar=np.std(calmars),
        mean_volatility=np.mean(volatilities),
        std_volatility=np.std(volatilities),
        # Efficiency
        mean_hit_rate=np.mean(hit_rates),
        std_hit_rate=np.std(hit_rates),
        mean_turnover=np.mean(turnovers),
        mean_costs=np.mean(costs),
    )


# =============================================================================
# Main Evaluator Class
# =============================================================================

class Evaluator:
    """
    Comprehensive evaluator for portfolio management agents.
    
    Runs all registered agents across multiple seeds and validation windows,
    producing publication-ready results tables and visualizations.
    
    Example
    -------
    >>> evaluator = Evaluator(EvaluationConfig())
    >>> evaluator.register_baseline("Equal Weight", create_equal_weight_agent)
    >>> evaluator.register_rl_agent("DQN", create_dqn_agent, checkpoint_path)
    >>> results = evaluator.run_evaluation(split="val")
    >>> evaluator.save_results(results)
    """
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Agent registries
        self._baselines: Dict[str, AgentFactory] = {}
        self._rl_agents: Dict[str, Tuple[AgentFactory, Path]] = {}
        
        # Load both dev and test datasets
        self.ds_dev = load_exported_dataset(config.dataset_path, split="dev")
        self.ds_test = load_exported_dataset(config.dataset_path, split="test")
        
        # Get validation windows from metadata
        self.validation_windows = self.ds_dev.metadata.get('validation_windows', [])
        
    def register_baseline(self, name: str, factory: AgentFactory):
        """Register a baseline agent factory."""
        self._baselines[name] = factory
        
    def register_rl_agent(self, name: str, factory: AgentFactory, checkpoint_path: Path):
        """Register an RL agent factory with its checkpoint."""
        self._rl_agents[name] = (factory, checkpoint_path)
    
    def _create_env(self, backend: DatasetBackend, seed: int, split: str = "val") -> PortfolioEnv:
        """Create environment with standard config."""
        env_config = EnvConfig(
            split=split,
            cost_rate=self.config.cost_rate,
            turnover_cap=self.config.turnover_cap,
            max_weight_per_asset=self.config.max_weight_per_asset,
            strict_projection=True,
            random_seed=seed,
        )
        return PortfolioEnv(env_config, backend)
    
    def _get_window_backends(self, split: str = "val") -> Dict[str, DatasetBackend]:
        """
        Get DatasetBackend for each validation window plus combined.
        
        Returns
        -------
        backends : Dict[str, DatasetBackend]
            Mapping window_name -> backend
        """
        backends = {}
        
        if split == "val":
            # Individual validation windows
            for window in self.validation_windows:
                window_name = window['name']
                tag = f"val_window_{window_name}"
                try:
                    backends[window_name] = DatasetBackend(self.ds_dev, split_tag_filter=[tag])
                except Exception as e:
                    print(f"Warning: Could not create backend for {window_name}: {e}")
            
            # Combined validation set
            all_tags = [f"val_window_{w['name']}" for w in self.validation_windows]
            backends["combined"] = DatasetBackend(self.ds_dev, split_tag_filter=all_tags)
            
        elif split == "test":
            # Test set - use full test dataset (2024-01-01 to 2025-10-31)
            # No split_tag_filter needed - use entire test set
            backends["test_full"] = DatasetBackend(self.ds_test)
        
        return backends
    
    def run_evaluation(self, split: str = "val", verbose: bool = True) -> Dict[str, List[AggregatedResult]]:
        """
        Run full evaluation across all agents, seeds, and windows.
        
        Parameters
        ----------
        split : str
            "val" for validation set, "test" for test set
        verbose : bool
            Print progress updates
            
        Returns
        -------
        results : Dict[str, List[AggregatedResult]]
            Mapping agent_name -> list of aggregated results (one per window)
        """
        if verbose:
            print("=" * 70)
            print(f"  EVALUATION: {split.upper()} SET")
            print(f"  Seeds: {self.config.seeds}")
            if split == "val":
                print(f"  Windows: {[w['name'] for w in self.validation_windows]} + combined")
            else:
                print(f"  Windows: test_full (646 days)")
            print("=" * 70)
        
        backends = self._get_window_backends(split)
        all_raw_results: List[AgentResult] = []
        aggregated_results: Dict[str, List[AggregatedResult]] = {}
        
        # Evaluate each agent
        all_agents = list(self._baselines.keys()) + list(self._rl_agents.keys())
        
        for agent_name in all_agents:
            if verbose:
                print(f"\n{'─' * 50}")
                print(f"  Evaluating: {agent_name}")
                print(f"{'─' * 50}")
            
            is_baseline = agent_name in self._baselines
            agent_type = "baseline" if is_baseline else "rl"
            
            aggregated_results[agent_name] = []
            
            # Evaluate on each window
            for window_name, backend in backends.items():
                window_results: List[AgentResult] = []
                
                if verbose:
                    print(f"  Window: {window_name} ({len(backend.dates())} days)")
                
                # Run across all seeds
                for seed in self.config.seeds:
                    env = self._create_env(backend, seed, split)
                    
                    # Create agent
                    if is_baseline:
                        factory = self._baselines[agent_name]
                        agent = factory(seed, env)
                    else:
                        factory, checkpoint = self._rl_agents[agent_name]
                        agent = factory(seed, env, checkpoint, self.config.device)
                    
                    # Run evaluation
                    result = run_single_evaluation(
                        agent=agent,
                        env=env,
                        seed=seed,
                        agent_name=agent_name,
                        agent_type=agent_type,
                        window_name=window_name,
                    )
                    
                    window_results.append(result)
                    all_raw_results.append(result)
                
                # Aggregate results for this window
                agg = aggregate_results(window_results)
                aggregated_results[agent_name].append(agg)
                
                if verbose:
                    print(f"    Return: {agg.mean_return*100:.2f}% ± {agg.std_return*100:.2f}%")
                    print(f"    95% CI: [{agg.ci_95_return[0]*100:.2f}%, {agg.ci_95_return[1]*100:.2f}%]")
                    print(f"    Sharpe: {agg.mean_sharpe:.3f} ± {agg.std_sharpe:.3f}")
        
        # Store raw results for detailed analysis
        self._raw_results = all_raw_results
        
        return aggregated_results
    
    def save_results(self, results: Dict[str, List[AggregatedResult]], prefix: str = ""):
        """
        Save results to CSV files.
        
        Creates:
        - {prefix}summary.csv: Aggregated results for all agents/windows
        - {prefix}raw_results.csv: Per-seed raw results
        - {prefix}per_window.csv: Per-window breakdown
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{prefix}_" if prefix else ""
        
        # 1. Summary table (publication-ready format)
        summary_rows = []
        for agent_name, agg_list in results.items():
            for agg in agg_list:
                summary_rows.append({
                    'Agent': agg.agent_name,
                    'Type': agg.agent_type,
                    'Window': agg.window_name,
                    'N_Runs': agg.n_runs,
                    # Profitability
                    'Return (%)': agg.mean_return * 100,
                    'Return Std (%)': agg.std_return * 100,
                    'Return CI Low (%)': agg.ci_95_return[0] * 100,
                    'Return CI High (%)': agg.ci_95_return[1] * 100,
                    'CAGR (%)': agg.mean_cagr * 100,
                    'CAGR Std (%)': agg.std_cagr * 100,
                    # Risk-adjusted
                    'Sharpe': agg.mean_sharpe,
                    'Sharpe Std': agg.std_sharpe,
                    'Sharpe CI Low': agg.ci_95_sharpe[0],
                    'Sharpe CI High': agg.ci_95_sharpe[1],
                    'Sortino': agg.mean_sortino,
                    'Sortino Std': agg.std_sortino,
                    'Calmar': agg.mean_calmar,
                    'Calmar Std': agg.std_calmar,
                    # Risk
                    'Max DD (%)': agg.mean_max_dd * 100,
                    'Max DD Std (%)': agg.std_max_dd * 100,
                    'Volatility (%)': agg.mean_volatility * 100,
                    'Volatility Std (%)': agg.std_volatility * 100,
                    # Efficiency
                    'Hit Rate (%)': agg.mean_hit_rate * 100,
                    'Hit Rate Std (%)': agg.std_hit_rate * 100,
                    'Turnover (%)': agg.mean_turnover * 100,
                    'Costs (%)': agg.mean_costs * 100,
                })
        
        summary_df = pd.DataFrame(summary_rows)
        summary_path = self.config.output_dir / f"{prefix}summary_{timestamp}.csv"
        summary_df.to_csv(summary_path, index=False, float_format='%.4f')
        print(f"\nSaved summary to: {summary_path}")
        
        # 2. Raw per-seed results
        raw_rows = [r.to_dict() for r in self._raw_results]
        raw_df = pd.DataFrame(raw_rows)
        raw_path = self.config.output_dir / f"{prefix}raw_results_{timestamp}.csv"
        raw_df.to_csv(raw_path, index=False, float_format='%.6f')
        print(f"Saved raw results to: {raw_path}")
        
        # 3. Per-window comparison (combined/test_full window, comparing all agents)
        combined_rows = []
        # Support both "combined" (val) and "test_full" (test)
        target_windows = ["combined", "test_full"]
        for agent_name, agg_list in results.items():
            combined_agg = [a for a in agg_list if a.window_name in target_windows]
            if combined_agg:
                agg = combined_agg[0]
                combined_rows.append({
                    'Agent': agg.agent_name,
                    'Return (%)': f"{agg.mean_return*100:.2f}",
                    'CAGR (%)': f"{agg.mean_cagr*100:.2f}",
                    '95% CI': f"[{agg.ci_95_return[0]*100:.2f}, {agg.ci_95_return[1]*100:.2f}]",
                    'Sharpe': f"{agg.mean_sharpe:.3f}",
                    'Sortino': f"{agg.mean_sortino:.3f}",
                    'Calmar': f"{agg.mean_calmar:.3f}",
                    'Max DD (%)': f"{agg.mean_max_dd*100:.2f}",
                    'Vol (%)': f"{agg.mean_volatility*100:.2f}",
                    'Hit Rate (%)': f"{agg.mean_hit_rate*100:.1f}",
                    'Turnover (%)': f"{agg.mean_turnover*100:.2f}",
                })
        
        combined_df = pd.DataFrame(combined_rows)
        window_label = "Test Set" if "test_full" in [a.window_name for agg_list in results.values() for a in agg_list] else "Combined Validation Set"
        print(f"\n{'=' * 120}")
        print(f"  SUMMARY: {window_label}")
        print("=" * 120)
        print(combined_df.to_string(index=False))
        
        return summary_path, raw_path
    
    def print_publication_table(self, results: Dict[str, List[AggregatedResult]]):
        """Print a publication-ready comparison table with all metrics."""
        print("\n" + "=" * 140)
        print("  PUBLICATION TABLE: Agent Comparison (Academic Metrics)")
        print("=" * 140)
        
        # Header Row 1: Main categories
        print(f"{'Agent':<18} │{'─── Profitability ───':^22}│{'──── Risk-Adjusted ────':^25}│{'───── Risk ─────':^20}│{'─── Efficiency ───':^18}")
        # Header Row 2: Column names
        print(f"{'':<18} │{'Return%':>8} {'CAGR%':>8} {'95%CI':>10}│{'Sharpe':>8} {'Sortino':>8} {'Calmar':>8}│{'MaxDD%':>8} {'Vol%':>8}│{'Hit%':>8} {'Turn%':>8}")
        print("─" * 140)
        
        # Support both "combined" (val) and "test_full" (test)
        target_windows = ["combined", "test_full"]
        
        # Get combined results for each agent
        for agent_name, agg_list in results.items():
            combined = [a for a in agg_list if a.window_name in target_windows]
            if combined:
                agg = combined[0]
                # Format CI as compact string
                ci_str = f"[{agg.ci_95_return[0]*100:+.1f},{agg.ci_95_return[1]*100:+.1f}]"
                print(f"{agg.agent_name:<18} │{agg.mean_return*100:>8.2f} {agg.mean_cagr*100:>8.2f} {ci_str:>10}│{agg.mean_sharpe:>8.3f} {agg.mean_sortino:>8.3f} {agg.mean_calmar:>8.3f}│{agg.mean_max_dd*100:>8.2f} {agg.mean_volatility*100:>8.2f}│{agg.mean_hit_rate*100:>8.1f} {agg.mean_turnover*100:>8.2f}")
        
        print("=" * 140)
        print("\nMetric definitions:")
        print("  Return%: Cumulative return | CAGR%: Compound Annual Growth Rate | Sharpe: Risk-adjusted return (Rf=0)")
        print("  Sortino: Downside-adjusted return | Calmar: Return/MaxDD | MaxDD%: Maximum drawdown")
        print("  Vol%: Annualized volatility | Hit%: % positive return days | Turn%: Mean portfolio turnover")
        print("=" * 140)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run evaluation with default configuration."""
    import argparse
    import torch
    
    parser = argparse.ArgumentParser(description="Evaluate portfolio agents")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"],
                       help="Dataset split to evaluate on")
    parser.add_argument("--output", type=str, default="results",
                       help="Output directory for results")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1024],
                       help="Random seeds for evaluation")
    args = parser.parse_args()
    
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    config = EvaluationConfig(
        seeds=args.seeds,
        output_dir=Path(args.output),
        device=device,
    )
    
    # Create evaluator
    evaluator = Evaluator(config)
    
    # Register baselines
    evaluator.register_baseline(
        "Equal Weight (1/N)",
        create_equal_weight_agent
    )
    
    evaluator.register_baseline(
        "Market Cap Weight",
        lambda seed, env: create_market_cap_agent(seed, env, config.max_weight_per_asset)
    )
    
    evaluator.register_baseline(
        "Mean-Variance (γ=1.0)",
        lambda seed, env: create_mean_variance_agent(seed, env, 1.0, config.max_weight_per_asset)
    )
    
    # Register RL agents
    dqn_checkpoint = Path("checkpoints/dqn_production/best")
    if (dqn_checkpoint / "dqn_checkpoint.pt").exists():
        evaluator.register_rl_agent(
            "DQN",
            lambda seed, env, ckpt, dev: create_dqn_agent(seed, env, ckpt, dev, use_double=False),
            dqn_checkpoint
        )
    else:
        print(f"Warning: DQN checkpoint not found at {dqn_checkpoint}")
    
    ddqn_checkpoint = Path("checkpoints/ddqn_production/best")
    if (ddqn_checkpoint / "dqn_checkpoint.pt").exists():
        evaluator.register_rl_agent(
            "Double DQN",
            lambda seed, env, ckpt, dev: create_dqn_agent(seed, env, ckpt, dev, use_double=True),
            ddqn_checkpoint
        )
    else:
        print(f"Warning: DDQN checkpoint not found at {ddqn_checkpoint}")
    
    # Run evaluation
    results = evaluator.run_evaluation(split=args.split, verbose=True)
    
    # Save and print results
    evaluator.save_results(results, prefix=args.split)
    evaluator.print_publication_table(results)


if __name__ == "__main__":
    main()
