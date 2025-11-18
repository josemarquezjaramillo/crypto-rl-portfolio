"""
Experience replay buffer for DQN.

This module implements a circular buffer for storing and sampling
transitions (s, a, r, s', done) during DQN training. Experience replay
is critical for breaking temporal correlations in financial time series
and stabilizing Q-learning.

Key Features:
- Circular buffer with fixed capacity (FIFO replacement)
- Efficient numpy-based storage
- Batch sampling for mini-batch gradient descent
- Handles variable observation dimensions via dictionary storage

Reference:
    Mnih et al. (2015) - "Human-level control through deep reinforcement learning"
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import numpy.typing as npt
from collections import deque


class ReplayBuffer:
    """
    Circular experience replay buffer for DQN.
    
    Stores transitions (obs, action, reward, next_obs, done) and provides
    random batch sampling for training. Uses deque for efficient FIFO
    management and separate storage for each component.
    
    Parameters
    ----------
    capacity : int
        Maximum number of transitions to store
    random_seed : int, optional
        Random seed for reproducible sampling
    
    Attributes
    ----------
    capacity : int
        Maximum buffer size
    size : int
        Current number of stored transitions
    
    Examples
    --------
    >>> buffer = ReplayBuffer(capacity=10000)
    >>> 
    >>> # Store transition
    >>> buffer.add(obs, action=5, reward=0.02, next_obs=next_obs, done=False)
    >>> 
    >>> # Sample batch
    >>> if len(buffer) >= 32:
    ...     batch = buffer.sample(32)
    ...     obs_batch, actions, rewards, next_obs_batch, dones = batch
    """
    
    def __init__(self, capacity: int, random_seed: Optional[int] = None):
        self.capacity = capacity
        self.rng = np.random.default_rng(random_seed)
        
        # Storage for each transition component
        # Using deque for efficient append/pop from both ends
        self.observations = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity)
        self.rewards = deque(maxlen=capacity)
        self.next_observations = deque(maxlen=capacity)
        self.dones = deque(maxlen=capacity)
    
    def add(self, 
            obs: Dict[str, Any], 
            action: int, 
            reward: float, 
            next_obs: Dict[str, Any], 
            done: bool) -> None:
        """
        Add a transition to the buffer.
        
        When buffer is full, oldest transition is automatically removed (FIFO).
        
        Parameters
        ----------
        obs : dict
            Current observation from environment
        action : int
            Catalog action index taken
        reward : float
            Reward received
        next_obs : dict
            Next observation after taking action
        done : bool
            Whether episode terminated
        """
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_observations.append(next_obs)
        self.dones.append(done)
    
    def sample(self, batch_size: int) -> Tuple[
        list,  # observations
        npt.NDArray[np.int64],  # actions
        npt.NDArray[np.float32],  # rewards
        list,  # next_observations
        npt.NDArray[np.bool_]  # dones
    ]:
        """
        Sample a random batch of transitions.
        
        Parameters
        ----------
        batch_size : int
            Number of transitions to sample
        
        Returns
        -------
        obs_batch : list of dict
            List of observations (length batch_size)
        actions_batch : np.ndarray
            Action indices [batch_size]
        rewards_batch : np.ndarray
            Rewards [batch_size], dtype float32
        next_obs_batch : list of dict
            List of next observations (length batch_size)
        dones_batch : np.ndarray
            Done flags [batch_size], dtype bool
        
        Raises
        ------
        ValueError
            If batch_size > current buffer size
        
        Notes
        -----
        Observations are returned as lists of dicts (not numpy arrays)
        because they have variable dimensions (A_t changes per sample).
        The Q-network must handle batch encoding internally.
        """
        if batch_size > len(self):
            raise ValueError(
                f"Cannot sample {batch_size} transitions from buffer "
                f"with only {len(self)} entries"
            )
        
        # Sample random indices
        indices = self.rng.choice(len(self), size=batch_size, replace=False)
        
        # Gather batch components
        obs_batch = [self.observations[i] for i in indices]
        actions_batch = np.array([self.actions[i] for i in indices], dtype=np.int64)
        rewards_batch = np.array([self.rewards[i] for i in indices], dtype=np.float32)
        next_obs_batch = [self.next_observations[i] for i in indices]
        dones_batch = np.array([self.dones[i] for i in indices], dtype=np.bool_)
        
        return obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch
    
    def __len__(self) -> int:
        """Return current number of transitions in buffer."""
        return len(self.observations)
    
    def is_ready(self, min_size: int) -> bool:
        """
        Check if buffer has enough transitions for training.
        
        Parameters
        ----------
        min_size : int
            Minimum required transitions
        
        Returns
        -------
        ready : bool
            True if buffer size >= min_size
        """
        return len(self) >= min_size
    
    def clear(self) -> None:
        """Clear all transitions from buffer."""
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_observations.clear()
        self.dones.clear()
    
    def __repr__(self) -> str:
        return f"ReplayBuffer(capacity={self.capacity}, size={len(self)})"
