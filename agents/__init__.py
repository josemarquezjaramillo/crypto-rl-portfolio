"""
Agent implementations for portfolio management RL.

This module provides base infrastructure and concrete agent implementations
for cryptocurrency portfolio management using reinforcement learning.

Available Agents
----------------
- BaseAgent: Abstract base class with common infrastructure
- (LinUCB, DQN, REINFORCE to be implemented in Weeks 2-4)

Usage
-----
>>> from agents.base_agent import BaseAgent, AgentConfig, EpisodeMetrics
>>> # Concrete agents will be imported as they're implemented
"""

from agents.base_agent import (
    BaseAgent,
    AgentConfig,
    EpisodeMetrics,
    MetricsTracker,
)

__all__ = [
    'BaseAgent',
    'AgentConfig',
    'EpisodeMetrics',
    'MetricsTracker',
]
