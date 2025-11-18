"""
Q-Network architecture for portfolio DQN.

This module implements the Q-network that maps observations to Q-values
for each catalog action. The network handles variable observation
dimensions (A_t changes over time) via pooling-based state encoding.

Key Design Decisions:
- Average pooling across assets to get fixed-size state representation
- Shallow architecture (2-3 layers) optimized for financial features
- Separate target network for stable TD learning
- Compatible with variable universe sizes A_t

Architecture inspired by:
    - Lucarelli & Borrotti (2020): Shallow networks for crypto portfolios
    - Mnih et al. (2015): Target network for DQN stability
"""

from typing import Dict, Any, List, Optional
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    Q-Network for portfolio management DQN.
    
    Maps variable-size observations to Q-values for each catalog action.
    Uses average pooling to handle changing number of tradable assets A_t.
    
    Architecture:
        Input: Pooled state representation [state_dim]
        Hidden: 2-3 fully connected layers with ReLU
        Output: Q-values for each catalog action [n_actions]
    
    Parameters
    ----------
    n_actions : int
        Number of catalog actions (portfolio strategies)
    state_dim : int
        Dimension of pooled state representation
    hidden_dims : List[int], optional
        Hidden layer dimensions (default: [512, 256])
    dropout : float, optional
        Dropout rate for regularization (default: 0.1)
    
    Attributes
    ----------
    fc_layers : nn.ModuleList
        Fully connected hidden layers
    output_layer : nn.Linear
        Final layer outputting Q-values
    
    Examples
    --------
    >>> q_net = QNetwork(n_actions=48, state_dim=256)
    >>> state = torch.randn(32, 256)  # Batch of 32 states
    >>> q_values = q_net(state)  # [32, 48]
    """
    
    def __init__(self, 
                 n_actions: int,
                 state_dim: int = 256,
                 hidden_dims: List[int] = [512, 256],
                 dropout: float = 0.1):
        super().__init__()
        
        self.n_actions = n_actions
        self.state_dim = state_dim
        
        # Build hidden layers
        layers = []
        in_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        self.fc_layers = nn.Sequential(*layers)
        
        # Output layer: Q-values for each action
        self.output_layer = nn.Linear(in_dim, n_actions)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/He initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Q-network.
        
        Parameters
        ----------
        state : torch.Tensor
            Batch of state representations [batch_size, state_dim]
        
        Returns
        -------
        q_values : torch.Tensor
            Q-values for each action [batch_size, n_actions]
        """
        hidden = self.fc_layers(state)
        q_values = self.output_layer(hidden)
        return q_values


class StateEncoder:
    """
    Encode variable-size observations into fixed-size state representations.
    
    Handles the challenge of variable A_t (tradable assets change daily)
    by pooling across the asset dimension. This allows the Q-network to
    operate on a fixed input size.
    
    Encoding Strategy:
        1. Flatten per-asset features: [A_t, 4, 60] → [A_t, 240]
        2. Concatenate with prev_weights: [A_t, 240] + [A_t] → [A_t, 241]
        3. Average pool across assets: [A_t, 241] → [241]
        4. Optional: Add global statistics (min, max, std) for richer representation
    
    Parameters
    ----------
    state_dim : int
        Output dimension of encoded state
    include_stats : bool, optional
        Whether to include min/max/std statistics (default: True)
    
    Attributes
    ----------
    state_dim : int
        Fixed output dimension
    
    Examples
    --------
    >>> encoder = StateEncoder(state_dim=256)
    >>> obs = {'features': np.random.randn(10, 4, 60),
    ...        'prev_weights': np.random.randn(10)}
    >>> state = encoder.encode(obs)  # [256]
    >>> 
    >>> # Batch encoding
    >>> obs_batch = [obs1, obs2, obs3]
    >>> states = encoder.encode_batch(obs_batch)  # [3, 256]
    """
    
    def __init__(self, state_dim: int = 256, include_stats: bool = True):
        self.state_dim = state_dim
        self.include_stats = include_stats
        self.device = 'cpu'  # Default device, can be set later
        
        # Per-asset feature dimension: 4 OHLCV channels × 60 days + 1 prev_weight = 241
        self.per_asset_dim = 4 * 60 + 1
        
        # Calculate feature dimensions
        if include_stats:
            # Pooled [241] + min [241] + max [241] + std [241] = 964
            self.raw_dim = self.per_asset_dim * 4
        else:
            # Just pooled [241]
            self.raw_dim = self.per_asset_dim
        
        # Linear projection to state_dim if needed
        if self.raw_dim != state_dim:
            self.projection = nn.Linear(self.raw_dim, state_dim)
            # Initialize weights
            nn.init.kaiming_normal_(self.projection.weight, nonlinearity='relu')
            nn.init.constant_(self.projection.bias, 0.0)
        else:
            self.projection = None
    
    def to(self, device: str) -> 'StateEncoder':
        """
        Move projection layer to specified device.
        
        Parameters
        ----------
        device : str
            Device to move to ('cpu' or 'cuda')
        
        Returns
        -------
        self : StateEncoder
            Returns self for method chaining
        """
        self.device = device
        if self.projection is not None:
            self.projection = self.projection.to(device)
        return self
    
    def encode(self, obs: Dict[str, Any]) -> npt.NDArray[np.float32]:
        """
        Encode single observation to fixed-size state.
        
        Parameters
        ----------
        obs : dict
            Observation with keys 'features' [A_t, 4, 60] and 'prev_weights' [A_t]
        
        Returns
        -------
        state : np.ndarray
            Encoded state [state_dim]
        """
        features = obs['features']  # [A_t, 4, 60]
        prev_weights = obs['prev_weights']  # [A_t]
        
        A_t = features.shape[0]
        
        # Flatten temporal dimension per asset
        features_flat = features.reshape(A_t, -1)  # [A_t, 240]
        
        # Concatenate with prev_weights
        asset_features = np.concatenate([
            features_flat,
            prev_weights[:, np.newaxis]
        ], axis=1)  # [A_t, 241]
        
        # Pooling across assets
        pooled = np.mean(asset_features, axis=0)  # [241]
        
        if self.include_stats:
            # Add statistical features
            min_features = np.min(asset_features, axis=0)  # [241]
            max_features = np.max(asset_features, axis=0)  # [241]
            std_features = np.std(asset_features, axis=0)  # [241]
            
            state_raw = np.concatenate([
                pooled, min_features, max_features, std_features
            ])  # [964]
        else:
            state_raw = pooled  # [241]
        
        # Project to target dimension if needed
        if self.projection is not None:
            state_tensor = torch.from_numpy(state_raw).float().unsqueeze(0).to(self.device)  # [1, raw_dim]
            with torch.no_grad():
                state = self.projection(state_tensor).squeeze(0).cpu().numpy()  # [state_dim]
        else:
            state = state_raw
        
        return state.astype(np.float32)
    
    def encode_batch(self, obs_batch: List[Dict[str, Any]]) -> npt.NDArray[np.float32]:
        """
        Encode batch of observations.
        
        Parameters
        ----------
        obs_batch : list of dict
            List of observations
        
        Returns
        -------
        states : np.ndarray
            Batch of encoded states [batch_size, state_dim]
        """
        states = np.stack([self.encode(obs) for obs in obs_batch], axis=0)
        return states
    
    def to_torch(self, states: npt.NDArray[np.float32], device: str = 'cpu') -> torch.Tensor:
        """
        Convert numpy states to torch tensors.
        
        Parameters
        ----------
        states : np.ndarray
            States [batch_size, state_dim] or [state_dim]
        device : str
            Device to place tensor on ('cpu' or 'cuda')
        
        Returns
        -------
        state_tensor : torch.Tensor
            States as torch tensor
        """
        return torch.from_numpy(states).float().to(device)


def copy_network_weights(source: nn.Module, target: nn.Module) -> None:
    """
    Copy weights from source network to target network (hard update).
    
    Used to update the target network in DQN. Target network is periodically
    synchronized with the online Q-network to stabilize learning.
    
    Parameters
    ----------
    source : nn.Module
        Source network (online Q-network)
    target : nn.Module
        Target network (target Q-network)
    
    Examples
    --------
    >>> q_net = QNetwork(n_actions=48)
    >>> target_net = QNetwork(n_actions=48)
    >>> copy_network_weights(q_net, target_net)
    """
    target.load_state_dict(source.state_dict())


def soft_update_network(source: nn.Module, target: nn.Module, tau: float = 0.001) -> None:
    """
    Soft update of target network (Polyak averaging).
    
    Alternative to hard updates: gradually blend source into target.
    target = tau * source + (1 - tau) * target
    
    Parameters
    ----------
    source : nn.Module
        Source network (online Q-network)
    target : nn.Module
        Target network (target Q-network)
    tau : float
        Blending factor (0 = no update, 1 = full copy)
    
    Examples
    --------
    >>> # Every training step, softly update target
    >>> soft_update_network(q_net, target_net, tau=0.001)
    """
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            tau * source_param.data + (1.0 - tau) * target_param.data
        )
