"""
Base agent infrastructure for portfolio management RL agents.

This module provides abstract base classes and utilities for implementing
portfolio management agents (LinUCB, DQN, REINFORCE). Design inspired by
Stable-Baselines3 with adaptations for financial portfolios.

Key Components:
- AgentConfig: Base configuration dataclass
- EpisodeMetrics: Structured episode performance data
- MetricsTracker: Lightweight metrics collection and aggregation
- BaseAgent: Abstract base class with common training/evaluation logic
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass, field, asdict
import numpy as np
import numpy.typing as npt
import csv
import json
from datetime import datetime

from environment.environment import PortfolioEnv, Obs


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class AgentConfig:
    """
    Base configuration for all agents.
    
    Attributes
    ----------
    name : str
        Agent identifier (e.g., "LinUCB", "DQN_alpha0.1")
    random_seed : int
        Random seed for reproducibility
    log_dir : Path, optional
        Directory for logs and checkpoints. If None, no logging occurs.
    checkpoint_freq : int
        Save checkpoint every N episodes
    """
    name: str
    random_seed: int = 42
    log_dir: Optional[Path] = None
    checkpoint_freq: int = 100


# ============================================================================
# Metrics
# ============================================================================

@dataclass
class EpisodeMetrics:
    """
    Performance metrics from a single episode.
    
    Captures both portfolio-level performance (returns, risk) and
    trading behavior (turnover, costs).
    """
    episode: int
    steps: int
    total_reward: float
    mean_reward_per_step: float
    final_portfolio_value: float
    cumulative_return: float  # (final_pv - 1.0)
    mean_turnover: float
    total_transaction_costs: float
    sharpe_ratio: float  # Annualized
    max_drawdown: float
    
    # Agent-specific metrics (e.g., DQN loss, LinUCB uncertainty)
    agent_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for easy logging/saving."""
        return asdict(self)


class MetricsTracker:
    """
    Lightweight tracker for agent performance metrics.
    
    Collects step-level data during episodes and computes summary
    statistics at episode end. Designed for minimal overhead.
    """
    
    def __init__(self):
        self.episodes: List[EpisodeMetrics] = []
        
        # Episode buffers (reset after each episode)
        self.current_episode_rewards: List[float] = []
        self.current_episode_turnovers: List[float] = []
        self.current_episode_costs: List[float] = []
        self.current_episode_pvs: List[float] = []
    
    def record_step(self, reward: float, info: Dict[str, Any]):
        """
        Record metrics from a single step.
        
        Parameters
        ----------
        reward : float
            Step reward
        info : dict
            Info dict from environment.step()
        """
        self.current_episode_rewards.append(reward)
        self.current_episode_turnovers.append(info.get('turnover', 0.0))
        self.current_episode_costs.append(info.get('transaction_cost', 0.0))
        self.current_episode_pvs.append(info.get('portfolio_value', 1.0))
    
    def end_episode(self, episode: int, agent_metrics: Optional[Dict[str, float]] = None) -> EpisodeMetrics:
        """
        Finalize episode and compute summary metrics.
        
        Parameters
        ----------
        episode : int
            Episode number
        agent_metrics : dict, optional
            Agent-specific metrics to include
        
        Returns
        -------
        metrics : EpisodeMetrics
            Computed episode metrics
        """
        n_steps = len(self.current_episode_rewards)
        total_reward = sum(self.current_episode_rewards)
        
        metrics = EpisodeMetrics(
            episode=episode,
            steps=n_steps,
            total_reward=total_reward,
            mean_reward_per_step=total_reward / n_steps if n_steps > 0 else 0.0,
            final_portfolio_value=self.current_episode_pvs[-1] if self.current_episode_pvs else 1.0,
            cumulative_return=self.current_episode_pvs[-1] - 1.0 if self.current_episode_pvs else 0.0,
            mean_turnover=np.mean(self.current_episode_turnovers) if self.current_episode_turnovers else 0.0,
            total_transaction_costs=sum(self.current_episode_costs),
            sharpe_ratio=self._compute_sharpe(),
            max_drawdown=self._compute_max_drawdown(),
            agent_metrics=agent_metrics or {}
        )
        
        self.episodes.append(metrics)
        self._reset_episode_buffers()
        return metrics
    
    def _compute_sharpe(self) -> float:
        """Compute annualized Sharpe ratio."""
        if len(self.current_episode_rewards) < 2:
            return 0.0
        
        mean_r = np.mean(self.current_episode_rewards)
        std_r = np.std(self.current_episode_rewards)
        
        if std_r < 1e-8:
            return 0.0
        
        sharpe = mean_r / std_r
        return sharpe * np.sqrt(365)  # Annualize for daily data
    
    def _compute_max_drawdown(self) -> float:
        """Compute maximum drawdown from portfolio values."""
        if not self.current_episode_pvs:
            return 0.0
        
        pvs = np.array(self.current_episode_pvs)
        running_max = np.maximum.accumulate(pvs)
        drawdown = (pvs - running_max) / (running_max + 1e-8)
        return float(np.min(drawdown))
    
    def _reset_episode_buffers(self):
        """Clear episode buffers."""
        self.current_episode_rewards = []
        self.current_episode_turnovers = []
        self.current_episode_costs = []
        self.current_episode_pvs = []
    
    def get_summary_stats(self, last_n: Optional[int] = None) -> Dict[str, float]:
        """
        Get summary statistics across episodes.
        
        Parameters
        ----------
        last_n : int, optional
            Only use last N episodes. If None, use all episodes.
        
        Returns
        -------
        stats : dict
            Summary statistics
        """
        if not self.episodes:
            return {}
        
        episodes = self.episodes[-last_n:] if last_n else self.episodes
        
        return {
            'n_episodes': len(episodes),
            'mean_episode_reward': np.mean([e.total_reward for e in episodes]),
            'std_episode_reward': np.std([e.total_reward for e in episodes]),
            'mean_reward_per_step': np.mean([e.mean_reward_per_step for e in episodes]),
            'mean_final_pv': np.mean([e.final_portfolio_value for e in episodes]),
            'mean_cumulative_return': np.mean([e.cumulative_return for e in episodes]),
            'mean_sharpe': np.mean([e.sharpe_ratio for e in episodes]),
            'mean_max_drawdown': np.mean([e.max_drawdown for e in episodes]),
            'mean_turnover': np.mean([e.mean_turnover for e in episodes]),
            'mean_transaction_costs': np.mean([e.total_transaction_costs for e in episodes]),
        }
    
    def to_dataframe(self):
        """
        Export all episode metrics as pandas DataFrame.
        
        Useful for Week 5 visualization and analysis.
        
        Returns
        -------
        df : pd.DataFrame
            DataFrame with one row per episode
        """
        import pandas as pd
        return pd.DataFrame([e.to_dict() for e in self.episodes])


# ============================================================================
# Base Agent
# ============================================================================

class BaseAgent(ABC):
    """
    Abstract base class for portfolio management agents.
    
    Provides common infrastructure for training, evaluation, logging,
    and checkpointing. All agents (LinUCB, DQN, REINFORCE) extend this
    class and implement agent-specific methods.
    
    Design inspired by Stable-Baselines3 with adaptations for:
    - Variable universe size (A_t changes over time)
    - Transaction costs and turnover constraints
    - Portfolio-specific metrics (Sharpe, drawdown, etc.)
    
    Abstract Methods (must implement)
    ---------------------------------
    - select_action(): Choose portfolio weights from observation
    - update(): Learn from experience
    - save()/load(): Model serialization
    
    Template Methods (can override)
    -------------------------------
    - on_episode_start(): Hook called at episode start
    - on_episode_end(): Hook called at episode end
    - get_agent_log_columns(): Define agent-specific CSV columns
    
    Examples
    --------
    See agents/linucb_agent.py, agents/dqn_agent.py for concrete implementations.
    """
    
    def __init__(self, config: AgentConfig, env: PortfolioEnv):
        self.config = config
        self.env = env
        self.rng = np.random.default_rng(config.random_seed)
        
        # Training state
        self.episode_count = 0
        self.step_count = 0
        
        # Metrics tracking
        self.metrics_tracker = MetricsTracker()
        
        # Logging setup
        self._agent_logger = None
        if config.log_dir:
            config.log_dir.mkdir(parents=True, exist_ok=True)
            self._agent_logger = self._init_agent_logger()
    
    # ============================================================
    # ABSTRACT METHODS (must implement in subclasses)
    # ============================================================
    
    @abstractmethod
    def select_action(self, obs: Obs, deterministic: bool = False) -> npt.NDArray[np.float32]:
        """
        Select portfolio weights given observation.
        
        Parameters
        ----------
        obs : Obs
            Current observation from environment
        deterministic : bool
            If True, use greedy/mean policy (for evaluation)
            If False, use exploration (for training)
        
        Returns
        -------
        action : np.ndarray, shape [A_t]
            Portfolio weights summing to 1.0
        """
        pass
    
    @abstractmethod
    def update(self, obs: Obs, action: npt.NDArray[np.float32], 
               reward: float, next_obs: Obs, done: bool) -> Optional[Dict[str, float]]:
        """
        Update agent from single experience.
        
        Parameters
        ----------
        obs : Obs
            Observation before action
        action : np.ndarray
            Action taken
        reward : float
            Reward received
        next_obs : Obs
            Observation after action
        done : bool
            Episode termination flag
        
        Returns
        -------
        metrics : dict or None
            Agent-specific training metrics (e.g., {'loss': 0.5}).
            Can return None if no update occurred (e.g., DQN before replay buffer fills).
        """
        pass
    
    @abstractmethod
    def save(self, path: Path) -> None:
        """
        Save agent state to disk.
        
        Should save:
        - Model parameters (networks, matrices, etc.)
        - Hyperparameters
        - Training state (episode count, etc.)
        
        Parameters
        ----------
        path : Path
            File path for checkpoint
        """
        pass
    
    @abstractmethod
    def load(self, path: Path) -> None:
        """
        Load agent state from disk.
        
        Parameters
        ----------
        path : Path
            File path to load from
        """
        pass
    
    # ============================================================
    # TRAINING (template method pattern)
    # ============================================================
    
    def train_episode(self) -> EpisodeMetrics:
        """
        Run one training episode.
        
        Template method that handles common logic (environment interaction,
        metrics tracking, checkpointing) and calls agent-specific hooks
        (on_episode_start, update, on_episode_end).
        
        Returns
        -------
        metrics : EpisodeMetrics
            Episode performance metrics
        """
        obs = self.env.reset()
        done = False
        
        # Episode-start hook (e.g., REINFORCE resets episode buffer)
        self.on_episode_start()
        
        # Collect agent metrics throughout episode
        episode_agent_metrics_list = []
        
        while not done:
            # Select action (agent-specific)
            action = self.select_action(obs, deterministic=False)
            
            # Step environment
            next_obs, reward, done, info = self.env.step(action)
            
            # Track metrics
            self.metrics_tracker.record_step(reward, info)
            
            # Update agent (agent-specific)
            agent_metrics = self.update(obs, action, reward, next_obs, done)
            if agent_metrics is not None:
                episode_agent_metrics_list.append(agent_metrics)
            
            obs = next_obs
            self.step_count += 1
        
        # Episode-end hook (e.g., REINFORCE does policy gradient update here)
        self.on_episode_end()
        
        # Aggregate agent metrics across episode
        aggregated_agent_metrics = {}
        if episode_agent_metrics_list:
            keys = episode_agent_metrics_list[0].keys()
            for key in keys:
                values = [m[key] for m in episode_agent_metrics_list if key in m]
                aggregated_agent_metrics[key] = np.mean(values) if values else 0.0
        
        # Finalize episode metrics
        self.episode_count += 1
        episode_metrics = self.metrics_tracker.end_episode(
            self.episode_count, 
            aggregated_agent_metrics
        )
        
        # Log and checkpoint
        self._log_episode(episode_metrics)
        
        if self.episode_count % self.config.checkpoint_freq == 0:
            self.checkpoint()
        
        return episode_metrics
    
    def on_episode_start(self):
        """
        Hook called at episode start.
        
        Override in subclass if needed (e.g., REINFORCE resets episode buffer).
        """
        pass
    
    def on_episode_end(self):
        """
        Hook called at episode end.
        
        Override in subclass if needed (e.g., REINFORCE computes returns and updates).
        """
        pass
    
    # ============================================================
    # EVALUATION
    # ============================================================
    
    def evaluate(self, n_episodes: int = 1, deterministic: bool = True) -> Dict[str, Any]:
        """
        Evaluate agent without training.
        
        Similar to Stable-Baselines3's evaluate_policy(). Runs episodes
        with deterministic policy and returns both summary statistics
        and episode-level details for plotting.
        
        Parameters
        ----------
        n_episodes : int
            Number of episodes to run
        deterministic : bool
            If True, use deterministic policy (default for evaluation)
        
        Returns
        -------
        results : dict
            Dictionary with keys:
            - 'summary': Aggregate statistics across episodes
            - 'episodes': List of EpisodeMetrics for each episode
        """
        temp_tracker = MetricsTracker()
        
        for ep in range(n_episodes):
            obs = self.env.reset()
            done = False
            
            while not done:
                action = self.select_action(obs, deterministic=deterministic)
                obs, reward, done, info = self.env.step(action)
                temp_tracker.record_step(reward, info)
            
            temp_tracker.end_episode(ep)
        
        return {
            'summary': temp_tracker.get_summary_stats(),
            'episodes': temp_tracker.episodes,  # For plotting
        }
    
    # ============================================================
    # UTILITIES FOR WEEK 5 COMPARISON & PLOTTING
    # ============================================================
    
    def get_training_history(self) -> Dict[str, List[float]]:
        """
        Get training metrics over time for plotting learning curves.
        
        Returns
        -------
        history : dict
            Dictionary with lists of metrics over episodes
        """
        if not self.metrics_tracker.episodes:
            return {}
        
        return {
            'episodes': [e.episode for e in self.metrics_tracker.episodes],
            'total_rewards': [e.total_reward for e in self.metrics_tracker.episodes],
            'sharpe_ratios': [e.sharpe_ratio for e in self.metrics_tracker.episodes],
            'portfolio_values': [e.final_portfolio_value for e in self.metrics_tracker.episodes],
            'turnovers': [e.mean_turnover for e in self.metrics_tracker.episodes],
            'max_drawdowns': [e.max_drawdown for e in self.metrics_tracker.episodes],
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get final performance summary for comparison table.
        
        Returns
        -------
        summary : dict
            Summary statistics for reporting
        """
        stats = self.metrics_tracker.get_summary_stats()
        stats['agent_name'] = self.config.name
        stats['total_episodes'] = self.episode_count
        stats['total_steps'] = self.step_count
        return stats
    
    # ============================================================
    # LOGGING & CHECKPOINTING
    # ============================================================
    
    def get_agent_log_columns(self) -> List[str]:
        """
        Override in subclass to add agent-specific log columns.
        
        Returns
        -------
        columns : list of str
            Column names for agent-specific metrics
        """
        return []  # No agent-specific columns by default
    
    def _init_agent_logger(self):
        """Initialize CSV logger for training metrics."""
        base_fieldnames = [
            'episode', 'step', 'total_reward', 'mean_reward_per_step',
            'sharpe', 'max_drawdown', 'mean_turnover', 'final_pv'
        ]
        
        # Add agent-specific columns
        agent_fieldnames = self.get_agent_log_columns()
        all_fieldnames = base_fieldnames + agent_fieldnames
        
        log_file = self.config.log_dir / f"{self.config.name}_training.csv"
        f = open(log_file, 'w', newline='')
        writer = csv.DictWriter(f, fieldnames=all_fieldnames)
        writer.writeheader()
        
        return {'file': f, 'writer': writer, 'fieldnames': all_fieldnames}
    
    def _log_episode(self, metrics: EpisodeMetrics):
        """Log episode metrics to CSV."""
        if not self._agent_logger:
            return
        
        row = {
            'episode': metrics.episode,
            'step': self.step_count,
            'total_reward': metrics.total_reward,
            'mean_reward_per_step': metrics.mean_reward_per_step,
            'sharpe': metrics.sharpe_ratio,
            'max_drawdown': metrics.max_drawdown,
            'mean_turnover': metrics.mean_turnover,
            'final_pv': metrics.final_portfolio_value,
        }
        
        # Add agent-specific metrics
        for col in self.get_agent_log_columns():
            row[col] = metrics.agent_metrics.get(col, 0.0)
        
        self._agent_logger['writer'].writerow(row)
        
        # Flush periodically
        if metrics.episode % 10 == 0:
            self._agent_logger['file'].flush()
    
    def checkpoint(self, suffix: str = ""):
        """
        Save checkpoint to log_dir.
        
        Parameters
        ----------
        suffix : str
            Optional suffix for checkpoint filename
        """
        if not self.config.log_dir:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.config.name}_ep{self.episode_count}"
        if suffix:
            filename += f"_{suffix}"
        
        # Save model
        model_path = self.config.log_dir / f"{filename}.pt"
        self.save(model_path)
        
        # Save metadata
        meta_path = self.config.log_dir / f"{filename}_meta.json"
        recent_episodes = self.metrics_tracker.episodes[-10:] if len(self.metrics_tracker.episodes) >= 10 else self.metrics_tracker.episodes
        
        with open(meta_path, 'w') as f:
            json.dump({
                'agent_name': self.config.name,
                'episode_count': self.episode_count,
                'step_count': self.step_count,
                'timestamp': timestamp,
                'recent_mean_reward': np.mean([e.total_reward for e in recent_episodes]) if recent_episodes else 0.0,
                'recent_mean_sharpe': np.mean([e.sharpe_ratio for e in recent_episodes]) if recent_episodes else 0.0,
            }, f, indent=2)
    
    def close(self):
        """Clean up resources (close log files, environment, etc.)."""
        if self._agent_logger:
            self._agent_logger['file'].close()
        self.env.close()
