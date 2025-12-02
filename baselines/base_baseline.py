"""
Base class for baseline portfolio strategies.

Provides common infrastructure for baseline agents that don't learn from
experience but still need to interact with the same environment and
constraint system as RL agents.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import numpy.typing as npt
import csv
import json
from datetime import datetime

from environment.environment import PortfolioEnv, Obs, StepInfo


@dataclass
class BaselineConfig:
    """
    Configuration for baseline agents.
    
    Attributes
    ----------
    name : str
        Agent identifier (e.g., "EqualWeight", "MarketCapWeight")
    random_seed : int
        Random seed for reproducibility (used in tie-breaking, etc.)
    log_dir : Path, optional
        Directory for logs. If None, no logging occurs.
    """
    name: str
    random_seed: int = 42
    log_dir: Optional[Path] = None


@dataclass
class BaselineEpisodeMetrics:
    """
    Performance metrics from a single episode.
    """
    episode: int
    steps: int
    total_reward: float
    mean_reward_per_step: float
    final_portfolio_value: float
    cumulative_return: float
    mean_turnover: float
    total_transaction_costs: float
    sharpe_ratio: float
    max_drawdown: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for logging."""
        return {
            'episode': self.episode,
            'steps': self.steps,
            'total_reward': self.total_reward,
            'mean_reward_per_step': self.mean_reward_per_step,
            'final_portfolio_value': self.final_portfolio_value,
            'cumulative_return': self.cumulative_return,
            'mean_turnover': self.mean_turnover,
            'total_transaction_costs': self.total_transaction_costs,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
        }


class BaselineAgent(ABC):
    """
    Abstract base class for baseline portfolio strategies.
    
    Baselines are non-learning agents that follow fixed allocation rules.
    They implement the same interface as RL agents for fair comparison.
    
    Subclasses must implement:
    - select_action(obs): Return target portfolio weights
    - get_name(): Return human-readable strategy name
    """
    
    def __init__(self, config: BaselineConfig):
        """
        Initialize baseline agent.
        
        Parameters
        ----------
        config : BaselineConfig
            Agent configuration
        """
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)
        
        # Episode tracking
        self._episode_rewards: List[float] = []
        self._episode_turnovers: List[float] = []
        self._episode_costs: List[float] = []
        self._episode_pvs: List[float] = []
        self._all_episodes: List[BaselineEpisodeMetrics] = []
    
    @abstractmethod
    def select_action(self, obs: Obs) -> npt.NDArray[np.float32]:
        """
        Select portfolio weights given current observation.
        
        Parameters
        ----------
        obs : Obs
            Current observation from environment containing:
            - features: [A_t, 4, 60] OHLCV tensor
            - prev_weights: [A_t] previous allocation
            - asset_ids: list of tradable assets
            - date: current decision date
        
        Returns
        -------
        weights : np.ndarray
            Target portfolio weights, shape [A_t], summing to 1.0
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_name(self) -> str:
        """Return human-readable strategy name."""
        raise NotImplementedError
    
    def update(self, obs: Obs, action: npt.NDArray[np.float32], 
               reward: float, next_obs: Obs, done: bool) -> Optional[Dict[str, float]]:
        """
        Baselines don't learn, so this is a no-op.
        
        Included for interface compatibility with RL agents.
        """
        return None
    
    def on_episode_start(self):
        """Reset episode tracking buffers."""
        self._episode_rewards = []
        self._episode_turnovers = []
        self._episode_costs = []
        self._episode_pvs = []
    
    def on_episode_end(self, episode: int) -> BaselineEpisodeMetrics:
        """
        Compute episode metrics at end of episode.
        
        Parameters
        ----------
        episode : int
            Episode number
            
        Returns
        -------
        metrics : BaselineEpisodeMetrics
            Computed performance metrics
        """
        n_steps = len(self._episode_rewards)
        total_reward = sum(self._episode_rewards)
        
        metrics = BaselineEpisodeMetrics(
            episode=episode,
            steps=n_steps,
            total_reward=total_reward,
            mean_reward_per_step=total_reward / n_steps if n_steps > 0 else 0.0,
            final_portfolio_value=self._episode_pvs[-1] if self._episode_pvs else 1.0,
            cumulative_return=self._episode_pvs[-1] - 1.0 if self._episode_pvs else 0.0,
            mean_turnover=np.mean(self._episode_turnovers) if self._episode_turnovers else 0.0,
            total_transaction_costs=sum(self._episode_costs),
            sharpe_ratio=self._compute_sharpe(),
            max_drawdown=self._compute_max_drawdown(),
        )
        
        self._all_episodes.append(metrics)
        return metrics
    
    def _compute_sharpe(self, annualization_factor: float = np.sqrt(365)) -> float:
        """
        Compute annualized Sharpe ratio from episode rewards.
        
        Uses daily rewards (which are log returns net of costs).
        Annualizes assuming 365 trading days (crypto).
        """
        if len(self._episode_rewards) < 2:
            return 0.0
        
        rewards = np.array(self._episode_rewards)
        mean_r = rewards.mean()
        std_r = rewards.std()
        
        if std_r < 1e-9:
            return 0.0
        
        return float(annualization_factor * mean_r / std_r)
    
    def _compute_max_drawdown(self) -> float:
        """
        Compute maximum drawdown from portfolio value series.
        """
        if len(self._episode_pvs) < 2:
            return 0.0
        
        pvs = np.array(self._episode_pvs)
        running_max = np.maximum.accumulate(pvs)
        drawdowns = (running_max - pvs) / running_max
        
        return float(drawdowns.max())
    
    def record_step(self, reward: float, info: StepInfo):
        """
        Record step metrics for episode tracking.
        
        Parameters
        ----------
        reward : float
            Step reward
        info : StepInfo
            Info dict from environment
        """
        self._episode_rewards.append(reward)
        self._episode_turnovers.append(info.get('turnover', 0.0))
        self._episode_costs.append(info.get('transaction_cost', 0.0))
        self._episode_pvs.append(info.get('portfolio_value', 1.0))
    
    def run_episode(self, env: PortfolioEnv, episode: int = 0) -> BaselineEpisodeMetrics:
        """
        Run a complete episode with this baseline strategy.
        
        Parameters
        ----------
        env : PortfolioEnv
            Environment to run in
        episode : int
            Episode number for logging
            
        Returns
        -------
        metrics : BaselineEpisodeMetrics
            Episode performance metrics
        """
        self.on_episode_start()
        
        obs = env.reset(seed=self.config.random_seed + episode)
        done = False
        
        while not done:
            action = self.select_action(obs)
            next_obs, reward, done, info = env.step(action)
            self.record_step(reward, info)
            obs = next_obs
        
        return self.on_episode_end(episode)
    
    def evaluate(self, env: PortfolioEnv, n_episodes: int = 1) -> Dict[str, Any]:
        """
        Evaluate baseline over multiple episodes.
        
        Parameters
        ----------
        env : PortfolioEnv
            Environment to evaluate in
        n_episodes : int
            Number of episodes to run
            
        Returns
        -------
        results : dict
            Aggregated evaluation results
        """
        episode_metrics = []
        
        for ep in range(n_episodes):
            metrics = self.run_episode(env, episode=ep)
            episode_metrics.append(metrics)
        
        # Aggregate results
        returns = [m.cumulative_return for m in episode_metrics]
        sharpes = [m.sharpe_ratio for m in episode_metrics]
        drawdowns = [m.max_drawdown for m in episode_metrics]
        turnovers = [m.mean_turnover for m in episode_metrics]
        costs = [m.total_transaction_costs for m in episode_metrics]
        
        return {
            'name': self.get_name(),
            'n_episodes': n_episodes,
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'mean_sharpe': np.mean(sharpes),
            'std_sharpe': np.std(sharpes),
            'mean_max_drawdown': np.mean(drawdowns),
            'mean_turnover': np.mean(turnovers),
            'mean_transaction_costs': np.mean(costs),
            'episodes': [m.to_dict() for m in episode_metrics],
        }
    
    def save(self, path: Path):
        """Save agent state (baselines have minimal state)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            'name': self.config.name,
            'random_seed': self.config.random_seed,
            'n_episodes_run': len(self._all_episodes),
        }
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load(self, path: Path):
        """Load agent state."""
        with open(path, 'r') as f:
            state = json.load(f)
        # Baselines have no learned parameters to restore
        return state
