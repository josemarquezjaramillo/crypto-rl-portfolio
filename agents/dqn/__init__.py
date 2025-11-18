"""
DQN (Deep Q-Network) agent for portfolio management.

This module implements a complete DQN agent with:
- Portfolio action catalog (47 discrete strategies)
- Experience replay buffer
- Q-network with target network
- State encoding for variable observations

Main exports:
- DQNAgent: Complete DQN agent class
- DQNConfig: Configuration dataclass
- PortfolioCatalog: Action catalog with 47 strategies
- ReplayBuffer: Experience replay implementation
- QNetwork: Q-network architecture
- StateEncoder: Variable observation encoder
"""

from agents.dqn.dqn_agent import DQNAgent, DQNConfig
from agents.dqn.action_catalog import PortfolioCatalog, Strategy
from agents.dqn.replay_buffer import ReplayBuffer
from agents.dqn.networks import QNetwork, StateEncoder, copy_network_weights

__all__ = [
    'DQNAgent',
    'DQNConfig',
    'PortfolioCatalog',
    'Strategy',
    'ReplayBuffer',
    'QNetwork',
    'StateEncoder',
    'copy_network_weights',
]
