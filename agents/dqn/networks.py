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
    
    Uses CANONICAL PADDING to preserve per-asset information:
    - Load all unique assets from dataset (dev + test)
    - Assign each asset a fixed position (alphabetical ordering)
    - Pad observations to fixed size with zeros for absent assets
    - Project to target dimension for Q-network input
    
    Key improvement over pooling: Q-network can now distinguish which assets 
    are held and learn asset-specific Q-values (e.g., "increase BTC" vs 
    "increase ETH"). This is critical for delta-based action catalogs where
    the same action has different feasibility depending on portfolio composition.
    
    Parameters
    ----------
    state_dim : int
        Output dimension of encoded state (default: 256)
    dataset_path : str
        Path to dataset directory for loading canonical asset list
    
    Attributes
    ----------
    state_dim : int
        Fixed output dimension
    canonical_assets : List[str]
        Alphabetically sorted list of all assets in dataset
    n_canonical : int
        Number of canonical asset positions
    raw_dim : int
        Dimension before projection (n_canonical × 240 + n_canonical)
    
    Examples
    --------
    >>> encoder = StateEncoder(state_dim=256, dataset_path="dataset_v1")
    >>> obs = {'features': np.random.randn(10, 4, 60),
    ...        'prev_weights': np.random.randn(10),
    ...        'asset_ids': ['bitcoin', 'ethereum', ...]}
    >>> state = encoder.encode(obs)  # [256]
    >>> 
    >>> # Batch encoding
    >>> obs_batch = [obs1, obs2, obs3]
    >>> states = encoder.encode_batch(obs_batch)  # [3, 256]
    """
    
    def __init__(self, state_dim: int = 256, dataset_path: str = "dataset_v1"):
        self.state_dim = state_dim
        self.dataset_path = dataset_path
        self.device = 'cpu'  # Will be set via .to(device)
        
        # Load canonical asset ordering from dataset
        self.canonical_assets = self._load_canonical_assets()
        self._asset_to_idx = {asset: i for i, asset in enumerate(self.canonical_assets)}
        
        # Compute dimensions
        self.n_canonical = len(self.canonical_assets)
        self.per_asset_features = 4 * 60  # OHLCV × lookback
        self.raw_dim = self.n_canonical * self.per_asset_features + self.n_canonical
        
        # Projection layer (always needed, raw_dim is large)
        self.projection = nn.Linear(self.raw_dim, state_dim)
        nn.init.kaiming_normal_(self.projection.weight, nonlinearity='relu')
        nn.init.constant_(self.projection.bias, 0.0)
    
    def _load_canonical_assets(self) -> List[str]:
        """
        Load all unique assets across dataset splits.
        
        Returns sorted (alphabetical) list for canonical ordering.
        Assets not seen during dev training will have zero features,
        which is equivalent to padding - no information leakage.
        
        Returns
        -------
        canonical_assets : List[str]
            Alphabetically sorted list of all unique assets
        """
        import json
        from pathlib import Path
        
        all_assets = set()
        dataset_dir = Path(self.dataset_path)
        
        # Load from all asset list files (dev and test)
        for asset_file in dataset_dir.glob("*_asset_lists.jsonl"):
            with open(asset_file) as f:
                for line in f:
                    data = json.loads(line)
                    all_assets.update(data['assets'])
        
        # Return alphabetically sorted for deterministic canonical ordering
        return sorted(all_assets)
    
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
        Encode single observation with canonical padding.
        
        Process:
        1. Create zero-padded arrays of size [n_canonical, 4, 60] and [n_canonical]
        2. Fill in positions for assets present in this observation
        3. Flatten and concatenate: [n_canonical × 240] + [n_canonical]
        4. Project to target dimension
        
        Assets appear at consistent positions across all observations:
        - bitcoin always at position X
        - ethereum always at position Y
        - etc. (alphabetical ordering)
        
        Parameters
        ----------
        obs : dict
            Observation with keys:
            - 'features': [A_t, 4, 60] OHLCV tensor
            - 'prev_weights': [A_t] portfolio weights
            - 'asset_ids': List[str] asset identifiers
        
        Returns
        -------
        state : np.ndarray
            Encoded state [state_dim]
        """
        features = obs['features']  # [A_t, 4, 60]
        prev_weights = obs['prev_weights']  # [A_t]
        asset_ids = obs['asset_ids']  # List[str], length A_t
        
        # Initialize padded arrays (zeros for absent assets)
        features_padded = np.zeros((self.n_canonical, 4, 60), dtype=np.float32)
        weights_padded = np.zeros(self.n_canonical, dtype=np.float32)
        
        # Fill canonical positions for assets present today
        for i, asset_id in enumerate(asset_ids):
            if asset_id in self._asset_to_idx:
                canonical_idx = self._asset_to_idx[asset_id]
                features_padded[canonical_idx] = features[i]
                weights_padded[canonical_idx] = prev_weights[i]
            # Note: If asset_id not in canonical list, skip (shouldn't happen)
        
        # Flatten features: [n_canonical, 4, 60] → [n_canonical × 240]
        features_flat = features_padded.reshape(-1)
        
        # Concatenate: [n_canonical × 240] + [n_canonical] → [raw_dim]
        state_raw = np.concatenate([features_flat, weights_padded])
        
        # Project to target dimension
        state_tensor = torch.from_numpy(state_raw).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            state = self.projection(state_tensor).squeeze(0).cpu().numpy()
        
        return state.astype(np.float32)
    
    def encode_batch(self, obs_batch: List[Dict[str, Any]]) -> npt.NDArray[np.float32]:
        """
        Encode batch of observations.
        
        With canonical padding, all states have same shape - straightforward stacking.
        
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
    
    def to_torch(self, states: npt.NDArray[np.float32], device: str = 'cuda') -> torch.Tensor:
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
