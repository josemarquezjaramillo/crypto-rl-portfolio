"""
DQN (Deep Q-Network) agent for portfolio management.

This module implements a complete DQN agent with:
- Delta-based action catalog (70 discrete rebalancing actions)
- Experience replay buffer
- Q-network with target network
- State encoding for variable observations

Main exports:
- DQNAgent: Complete DQN agent class
- DQNConfig: Configuration dataclass
- DeltaActionCatalog: Action catalog with 70 delta actions
- ReplayBuffer: Experience replay implementation
- QNetwork: Q-network architecture
- StateEncoder: Variable observation encoder

Legacy exports (deprecated):
- PortfolioCatalog: Old fixed-strategy catalog (48 actions) - use action_catalog_legacy.py
- Strategy: Legacy action wrapper - use DeltaAction instead
"""

from agents.dqn.dqn_agent import DQNAgent, DQNConfig
from agents.dqn.action_catalog_delta import DeltaActionCatalog, DeltaAction
from agents.dqn.replay_buffer import ReplayBuffer
from agents.dqn.networks import QNetwork, StateEncoder, copy_network_weights

__all__ = [
    'DQNAgent',
    'DQNConfig',
    'DeltaActionCatalog',
    'DeltaAction',
    'ReplayBuffer',
    'QNetwork',
    'StateEncoder',
    'copy_network_weights',
]
