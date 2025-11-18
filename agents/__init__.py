"""
Agent implementations for portfolio management RL.

This module provides base infrastructure and concrete agent implementations
for cryptocurrency portfolio management using reinforcement learning.

Available Agents
----------------
- BaseAgent: Abstract base class with common infrastructure
- DQNAgent: Deep Q-Network for portfolio allocation (Week 3)
- (REINFORCE, LinUCB to be implemented in Week 4)

Usage
-----
>>> from agents.base_agent import BaseAgent, AgentConfig, EpisodeMetrics
>>> from agents.dqn import DQNAgent, DQNConfig
>>> 
>>> # Configure and create DQN agent
>>> config = DQNConfig(name="DQN_v1", buffer_size=10000, learning_rate=1e-4)
>>> agent = DQNAgent(config, env)
"""

from agents.base_agent import (
    BaseAgent,
    AgentConfig,
    EpisodeMetrics,
    MetricsTracker,
)

# Import DQN components (Week 3)
from agents.dqn import DQNAgent, DQNConfig

__all__ = [
    'BaseAgent',
    'AgentConfig',
    'EpisodeMetrics',
    'MetricsTracker',
    'DQNAgent',
    'DQNConfig',
]
