"""
Deep Q-Network (DQN) agent for portfolio management.

This module implements a DQN agent that learns to select portfolio rebalancing
actions from a discrete catalog of delta-based adjustments. The agent uses
experience replay and target networks to stabilize learning in the nonstationary
crypto market.

Key Components:
- Delta action catalog (70 rebalancing actions: adjust, rotate, diversify, etc.)
- Experience replay buffer (breaks temporal correlations)
- Q-network with target network (stable TD learning)
- ε-greedy exploration (exploration vs exploitation)

Design follows:
    - Lucarelli & Borrotti (2020): DQN for crypto trading with delta actions
    - Mnih et al. (2015): Experience replay + target networks
    - BaseAgent infrastructure: Common training/evaluation logic

Usage Example:
    >>> from pathlib import Path
    >>> from agents.dqn import DQNAgent, DQNConfig
    >>> from environment.environment import PortfolioEnv
    >>> 
    >>> # Configure DQN
    >>> config = DQNConfig(
    ...     name="DQN_v1",
    ...     buffer_size=10000,
    ...     batch_size=64,
    ...     gamma=0.99,
    ...     epsilon_start=1.0,
    ...     epsilon_end=0.1,
    ...     epsilon_decay_episodes=500,
    ...     target_update_freq=1000,
    ...     learning_rate=1e-4,
    ...     log_dir=Path("logs/dqn"),
    ... )
    >>> 
    >>> # Create agent
    >>> agent = DQNAgent(config, env)
    >>> 
    >>> # Train
    >>> agent.train(n_episodes=1000)
    >>> 
    >>> # Evaluate
    >>> results = agent.evaluate(n_episodes=10, deterministic=True)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.optim as optim

from agents.base_agent import BaseAgent, AgentConfig
from agents.dqn.action_catalog_delta import DeltaActionCatalog
from agents.dqn.replay_buffer import ReplayBuffer
from agents.dqn.networks import QNetwork, StateEncoder, copy_network_weights
from environment.environment import PortfolioEnv, Obs


@dataclass
class DQNConfig(AgentConfig):
    """
    Configuration for DQN agent.
    
    Extends base AgentConfig with DQN-specific hyperparameters.
    
    Attributes
    ----------
    buffer_size : int
        Replay buffer capacity
    batch_size : int
        Mini-batch size for training
    gamma : float
        Discount factor for TD learning
    epsilon_start : float
        Initial exploration rate
    epsilon_end : float
        Final exploration rate
    epsilon_decay_episodes : int
        Number of episodes to decay epsilon from start to end
    target_update_freq : int
        Update target network every N steps
    learning_rate : float
        Learning rate for Q-network optimizer
    state_dim : int
        Dimension of encoded state representation
    hidden_dims : List[int]
        Hidden layer dimensions for Q-network
    dropout : float
        Dropout rate for Q-network
    min_buffer_size : int
        Minimum buffer size before training starts
    device : str
        Device for PyTorch ('cpu' or 'cuda')
    """
    # Replay buffer
    buffer_size: int = 10000
    batch_size: int = 64
    min_buffer_size: int = 1000
    
    # Q-learning
    gamma: float = 0.99
    learning_rate: float = 1e-4
    
    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay_episodes: int = 500
    
    # Target network
    target_update_freq: int = 1000
    
    # Network architecture
    state_dim: int = 256
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256])
    dropout: float = 0.1
    
    # Device
    device: str = "cuda"  # Change to "cuda" if GPU available


class DQNAgent(BaseAgent):
    """
    Deep Q-Network agent for portfolio management.
    
    Learns to select portfolio rebalancing actions from a discrete catalog of
    delta-based adjustments by estimating Q-values (expected returns) for each
    action given the current market state and previous portfolio allocation.
    
    Parameters
    ----------
    config : DQNConfig
        Agent configuration
    env : PortfolioEnv
        Portfolio environment
    
    Attributes
    ----------
    catalog : DeltaActionCatalog
        Discrete action space (delta-based rebalancing actions)
    replay_buffer : ReplayBuffer
        Experience replay buffer
    q_network : QNetwork
        Online Q-network
    target_network : QNetwork
        Target Q-network (for stable TD targets)
    state_encoder : StateEncoder
        Encodes variable observations to fixed-size states
    optimizer : torch.optim.Optimizer
        Optimizer for Q-network
    epsilon : float
        Current exploration rate
    """
    
    def __init__(self, config: DQNConfig, env: PortfolioEnv):
        super().__init__(config, env)
        
        self.config: DQNConfig = config  # Type hint for IDE
        
        # Delta action catalog (70 rebalancing actions)
        self.catalog = DeltaActionCatalog()
        self.n_actions = self.catalog.size
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=config.buffer_size,
            random_seed=config.random_seed
        )
        
        # State encoder (move to device to ensure projection layer is on correct device)
        self.state_encoder = StateEncoder(state_dim=config.state_dim).to(config.device)
        
        # Q-networks
        self.q_network = QNetwork(
            n_actions=self.n_actions,
            state_dim=config.state_dim,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout
        ).to(config.device)
        
        self.target_network = QNetwork(
            n_actions=self.n_actions,
            state_dim=config.state_dim,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout
        ).to(config.device)
        
        # Initialize target network to match Q-network
        copy_network_weights(self.q_network, self.target_network)
        self.target_network.eval()  # Target network always in eval mode
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=config.learning_rate
        )
        
        # Exploration schedule
        self.epsilon = config.epsilon_start
        self.epsilon_decay_per_episode = (
            (config.epsilon_start - config.epsilon_end) / config.epsilon_decay_episodes
        )
        
        # Training state
        self.update_count = 0  # Number of Q-network updates
        self.last_action_idx = None  # For logging
        self.last_q_values = None  # For logging
    
    def select_action(self, obs: Obs, deterministic: bool = False) -> npt.NDArray[np.float32]:
        """
        Select portfolio weights using ε-greedy policy.
        
        Parameters
        ----------
        obs : Obs
            Current observation from environment
        deterministic : bool
            If True, use greedy policy (no exploration)
        
        Returns
        -------
        weights : np.ndarray
            Portfolio weights [A_t]
        """
        # Encode observation to state
        state = self.state_encoder.encode(obs)
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.config.device)
        
        # Get Q-values
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state_tensor).squeeze(0)  # [n_actions]
        self.q_network.train()
        
        # ε-greedy action selection
        if not deterministic and self.rng.random() < self.epsilon:
            # Explore: random action
            action_idx = self.rng.integers(0, self.n_actions)
        else:
            # Exploit: greedy action
            action_idx = q_values.argmax().item()
        
        # Store for logging
        self.last_action_idx = action_idx
        self.last_q_values = q_values.cpu().numpy()
        
        # Apply delta action to previous weights
        prev_weights = obs['prev_weights']
        weights = self.catalog.apply_action(action_idx, obs, prev_weights)
        
        return weights
    
    def update(self, 
               obs: Obs, 
               action: npt.NDArray, 
               reward: float, 
               next_obs: Obs, 
               done: bool) -> Optional[Dict[str, float]]:
        """
        Update Q-network using experience replay and TD learning.
        
        Parameters
        ----------
        obs : Obs
            Previous observation
        action : np.ndarray
            Action taken (portfolio weights, not used - we use last_action_idx)
        reward : float
            Reward received
        next_obs : Obs
            Next observation
        done : bool
            Whether episode terminated
        
        Returns
        -------
        metrics : dict or None
            Training metrics (loss, Q-value stats) if update performed
        """
        # Add transition to replay buffer
        self.replay_buffer.add(obs, self.last_action_idx, reward, next_obs, done)
        
        # Wait until buffer has enough samples
        if not self.replay_buffer.is_ready(self.config.min_buffer_size):
            return None
        
        # Sample mini-batch
        obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch = \
            self.replay_buffer.sample(self.config.batch_size)
        
        # Encode states
        states = self.state_encoder.encode_batch(obs_batch)
        next_states = self.state_encoder.encode_batch(next_obs_batch)
        
        # Convert to tensors
        states_tensor = torch.from_numpy(states).float().to(self.config.device)
        next_states_tensor = torch.from_numpy(next_states).float().to(self.config.device)
        actions_tensor = torch.from_numpy(actions_batch).long().to(self.config.device)
        rewards_tensor = torch.from_numpy(rewards_batch).float().to(self.config.device)
        dones_tensor = torch.from_numpy(dones_batch).float().to(self.config.device)
        
        # Compute current Q-values: Q(s, a)
        current_q_values = self.q_network(states_tensor)  # [batch_size, n_actions]
        current_q = current_q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)  # [batch_size]
        
        # Compute target Q-values: r + γ * max_a' Q_target(s', a')
        with torch.no_grad():
            next_q_values = self.target_network(next_states_tensor)  # [batch_size, n_actions]
            next_q = next_q_values.max(1)[0]  # [batch_size]
            target_q = rewards_tensor + (1.0 - dones_tensor) * self.config.gamma * next_q
        
        # Compute TD loss
        loss = nn.functional.mse_loss(current_q, target_q)
        
        # Optimize Q-network
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to stabilize training
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        # Update target network periodically
        self.update_count += 1
        if self.update_count % self.config.target_update_freq == 0:
            copy_network_weights(self.q_network, self.target_network)
        
        # Return metrics
        return {
            'td_loss': loss.item(),
            'q_mean': current_q.mean().item(),
            'q_std': current_q.std().item(),
            'target_q_mean': target_q.mean().item(),
        }
    
    def on_episode_end(self) -> None:
        """
        Hook called at end of each episode.
        
        Decays epsilon (exploration rate) according to schedule.
        """
        super().on_episode_end()
        
        # Decay epsilon
        if self.epsilon > self.config.epsilon_end:
            self.epsilon = max(
                self.config.epsilon_end,
                self.epsilon - self.epsilon_decay_per_episode
            )
    
    def save(self, path: Path) -> None:
        """
        Save agent checkpoint.
        
        Parameters
        ----------
        path : Path
            Directory to save checkpoint
        """
        path.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'config': self.config,
            'q_network_state': self.q_network.state_dict(),
            'target_network_state': self.target_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'step_count': self.step_count,
            'update_count': self.update_count,
        }
        
        torch.save(checkpoint, path / 'dqn_checkpoint.pt')
        
        # Save metrics history
        if self.metrics_tracker.episodes:
            self.metrics_tracker.to_csv(path / 'training_history.csv')
    
    def load(self, path: Path) -> None:
        """
        Load agent checkpoint.
        
        Parameters
        ----------
        path : Path
            Directory containing checkpoint
        """
        checkpoint_path = path / 'dqn_checkpoint.pt'
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        
        # Restore networks
        self.q_network.load_state_dict(checkpoint['q_network_state'])
        self.target_network.load_state_dict(checkpoint['target_network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        # Restore training state
        self.epsilon = checkpoint['epsilon']
        self.episode_count = checkpoint['episode_count']
        self.step_count = checkpoint['step_count']
        self.update_count = checkpoint['update_count']
        
        print(f"Loaded checkpoint from {path}")
        print(f"  Episode: {self.episode_count}")
        print(f"  Steps: {self.step_count}")
        print(f"  Epsilon: {self.epsilon:.3f}")
    
    def get_agent_log_columns(self) -> List[str]:
        """
        Define DQN-specific log columns.
        
        Returns
        -------
        columns : list of str
            Column names for CSV logging
        """
        return [
            'epsilon',
            'buffer_size',
            'td_loss',
            'q_mean',
            'q_std',
            'target_q_mean',
            'action_idx',
            'strategy_name',
        ]
    
    def _get_agent_metrics(self) -> Dict[str, Any]:
        """
        Get current agent-specific metrics for logging.
        
        Returns
        -------
        metrics : dict
            DQN-specific metrics
        """
        metrics = {
            'epsilon': self.epsilon,
            'buffer_size': len(self.replay_buffer),
            'action_idx': self.last_action_idx if self.last_action_idx is not None else -1,
        }
        
        if self.last_action_idx is not None:
            metrics['strategy_name'] = self.catalog.get_strategy_name(self.last_action_idx)
        else:
            metrics['strategy_name'] = 'N/A'
        
        return metrics
    
    def __repr__(self) -> str:
        return (
            f"DQNAgent(name='{self.config.name}', "
            f"n_actions={self.n_actions}, "
            f"buffer_size={len(self.replay_buffer)}/{self.config.buffer_size}, "
            f"epsilon={self.epsilon:.3f})"
        )
