import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Dirichlet

import numpy as np
import numpy.typing as npt
from typing import List, Dict, Any, Tuple
from pathlib import Path
import json

from pathlib import Path

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.optim as optim

from agents.base_agent import BaseAgent, AgentConfig
# from agents.dqn.action_catalog_delta import DeltaActionCatalog
# from agents.dqn.replay_buffer import ReplayBuffer
# from agents.dqn.networks import QNetwork, StateEncoder, copy_network_weights
from environment.environment import PortfolioEnv, Obs

# from data.dataset_loader import load_exported_dataset
# from data.dataset_backend import DatasetBackend
# from environment.environment import PortfolioEnv, EnvConfig
# from agents.base_agent import *
# from agents.dqn.networks import StateEncoder

import sys
from pathlib import Path
import argparse
import numpy as np
import torch
# import optuna
from datetime import datetime
from dotenv import load_dotenv
import json

from tqdm.auto import tqdm

# Load environment variables
load_dotenv()

# Add project root to path
# project_root = Path(__file__).parent.parent.parent
# sys.path.insert(0, str(project_root))

# from data.data_loader import DatabaseConfig
# from data.dataset_loader import load_exported_dataset
# from data.dataset_backend import DatasetBackend
# from environment.environment import PortfolioEnv, EnvConfig


class AssetIndexer:
    """
    Canonical asset reindexer + feature/weight organizer with mask.

    Same constructor semantics as StateEncoder:
      __init__(state_dim=256, dataset_path="dataset_v1")

    Given an observation with:
      features:     [A_t, 4, 60]
      prev_weights: [A_t]
      asset_ids:    list[A_t]

    This class outputs:
      features_out : [n_canonical, 60, 4]   (canonical padded, transposed for GRU)
      weights_out  : [n_canonical]
      mask_out     : [n_canonical] bool     (True where asset is present in obs)
    """

    def __init__(self, state_dim: int = 256, dataset_path: str = "dataset_v1"):
        self.state_dim = state_dim          # kept for API compatibility, not used
        self.dataset_path = dataset_path

        # Load canonical asset ordering from dataset (same as StateEncoder)
        self.canonical_assets = self._load_canonical_assets()
        self._asset_to_idx = {asset: i for i, asset in enumerate(self.canonical_assets)}

        self.n_canonical = len(self.canonical_assets)
        self.per_asset_features = 4 * 60  # OHLCV × lookback, same as StateEncoder

    def _load_canonical_assets(self) -> List[str]:
        """
        Load all unique assets across dataset splits.

        Same logic as StateEncoder._load_canonical_assets.
        """
        all_assets = set()
        dataset_dir = Path(self.dataset_path)

        # Load from all asset list files (dev and test)
        for asset_file in dataset_dir.glob("*_asset_lists.jsonl"):
            with open(asset_file) as f:
                for line in f:
                    data = json.loads(line)
                    all_assets.update(data["assets"])

        # Alphabetical ordering for determinism
        return sorted(all_assets)

    def reindex(
        self, obs: Dict[str, Any]
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
        """
        Canonically pad + reindex a single observation.

        Input obs:
          obs['features']     : [A_t, 4, 60]
          obs['prev_weights'] : [A_t]
          obs['asset_ids']    : list[str] length A_t

        Returns:
          features_out : [n_canonical, 60, 4]
          weights_out  : [n_canonical]
          mask_out     : [n_canonical] bool
        """
        features = obs["features"]      # [A_t, 4, 60]
        prev_weights = obs["prev_weights"]  # [A_t]
        asset_ids = obs["asset_ids"]    # list[str], length A_t

        # Initialize padded arrays (zeros for absent assets), same as StateEncoder
        features_padded = np.zeros((self.n_canonical, 4, 60), dtype=np.float32)
        weights_padded = np.zeros(self.n_canonical, dtype=np.float32)
        mask = np.zeros(self.n_canonical, dtype=bool)

        # Fill canonical positions for assets present today
        for i, asset_id in enumerate(asset_ids):
            idx = self._asset_to_idx.get(asset_id, None)
            if idx is None:
                # Unknown asset – skip (shouldn't happen if dataset + env are aligned)
                continue
            features_padded[idx] = features[i]          # [4, 60]
            weights_padded[idx] = prev_weights[i]
            mask[idx] = True

        # For your GRU: transpose per-asset from [4, 60] → [60, 4]
        # Result: [n_canonical, 60, 4]
        features_out = np.transpose(features_padded, (0, 2, 1))

        return features_out, weights_padded, mask


class PolicyNet(nn.Module):
    """
    Simplest functional policy network:
      GRU → per-asset embedding → concat prev_w + no_prev → MLP → logits

    Input shapes:
      features:     [A_t, 60, 4]
      prev_weights: [A_t]
      no_prev:      bool

    Output:
      logits:       [A_t]
    """

    def __init__(self, feature_dim=4, hidden_dim=64, seq_layer=nn.GRU):
        super().__init__()

        self.hidden_dim = hidden_dim

        # GRU for each asset's sequence
        self.encoder = seq_layer(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        # MLP after concatenating prev_w + no_prev flag
        # asset_emb (hidden_dim) + prev_w(1) + no_prev(1) = hidden_dim + 2
        self.fc1 = nn.Linear(hidden_dim + 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, features, prev_weights, no_prev):
        """
        features:     [A_t, 60, 4]
        prev_weights: [A_t]
        no_prev:      bool
        """

        device = features.device
        A_t = features.size(0)

        # ----------- 1. Encode sequences ----------- #
        # GRU expects [batch=A_t, seq=60, feature=4]
        enc, _ = self.encoder(features)        # [A_t, 60, hidden]
        asset_emb = enc[:, -1]                 # last timestep: [A_t, hidden]

        # ----------- 2. Prepare prev weight inputs ----------- #
        pw = prev_weights.to(device).unsqueeze(1)   # [A_t, 1]

        no_prev_flag = torch.full(
            (A_t, 1),
            1.0 if no_prev else 0.0,
            device=device,
            dtype=pw.dtype
        )                                            # [A_t, 1]

        # ----------- 3. Combine features ----------- #
        x = torch.cat([asset_emb, pw, no_prev_flag], dim=1)  # [A_t, hidden+2]

        # ----------- 4. Feedforward head ----------- #
        x = torch.relu(self.fc1(x))                 # [A_t, hidden]
        logits = self.fc2(x).squeeze(-1)            # [A_t]

        return logits

@dataclass
class PolicyGradConfig(AgentConfig):
  # learning
  learning_rate=1e-4
  epsilon_start=1.0
  epsilon_end = 0.05
  epsilon_decay_episodes = 500
  # state encoding
  dataset_path="dataset_v1"
  max_alloc=0.35
  # pytorch
  device='cuda'
  recurrent_layer = nn.GRU
  #architecture
  layers=None
  hidden_dim = 64
  # canonical_assets: List[str] = field(default_factory=list)

class PolicyGradAgent(BaseAgent):
    def __init__(self, config, env):
        super().__init__(config, env)

        self.config = config
        self.device = config.device
        self.max_alloc = config.max_alloc

        # Epsilon-greedy exploration schedule (used to occasionally ignore policy)
        self.epsilon_decay_episodes = config.epsilon_decay_episodes
        self.epsilon = config.epsilon_start
        self.epsilon_decay_per_episode = (
            config.epsilon_start - config.epsilon_end
        ) / self.epsilon_decay_episodes

        # Canonical asset indexer
        self.asset_indexer = AssetIndexer(
            dataset_path=config.dataset_path
        )

        # Policy network → outputs logits for active assets
        self.nn = PolicyNet(
            feature_dim=4,
            hidden_dim=config.hidden_dim,
            seq_layer = config.recurrent_layer
        ).to(self.device)

        # Training
        self.training = True
        self.optimizer = torch.optim.Adam(self.nn.parameters(), lr=config.learning_rate)

        # Episode-wise bookkeeping
        self.episode_start = True        # flag for "no_prev" mask into PolicyNet
        self.logprobs = []               # list[Tensor], each scalar log π(a_t | s_t)
        self.rewards_buf = []            # list[float], env rewards per step
        self.det_step = False            # tracking whether last action used policy/logprob

    # ------------------------------------------------------------------
    # RANDOM POLICY (for epsilon exploration)
    # ------------------------------------------------------------------
    def random_policy(self, obs=None):
        """
        Sample a random valid portfolio over the *active* assets.
        """
        if obs is None:
            raise ValueError("random_policy() requires obs for correct asset count.")

        A_t = len(obs['asset_ids'])
        if A_t == 0:
            return np.array([], dtype=np.float32)

        w = np.random.rand(A_t).astype(np.float32)
        w /= w.sum()
        return w

    # ------------------------------------------------------------------
    # ACTION SELECTION
    # ------------------------------------------------------------------
    def select_action(self, obs, deterministic: bool = False):
        """
        Select an action and (if training) store log_prob for REINFORCE.
        For training (deterministic=False), we:
          - compute Dirichlet concentration params from PolicyNet
          - sample weights from Dirichlet
          - store log_prob for the REINFORCE update
        """
        # ----- epsilon-greedy exploration: ignore policy sometimes -----
        if (not deterministic) and (self.rng.random() < self.epsilon):
            self.det_step = False
            return self.random_policy(obs)

        # ----- policy-driven action -----
        # Canonical indexing -> then slice to active assets
        asset_features, prev_weights, mask = self.asset_indexer.reindex(obs)

        asset_features = torch.tensor(asset_features, dtype=torch.float32, device=self.device)
        prev_weights   = torch.tensor(prev_weights,   dtype=torch.float32, device=self.device)
        mask_t         = torch.tensor(mask,           dtype=torch.bool,    device=self.device)

        # Only keep active assets
        features_active = asset_features[mask_t]     # [A_t, 60, 4]
        prev_w_active   = prev_weights[mask_t]       # [A_t]

        # Forward: logits for each active asset
        logits = self.nn(features_active, prev_w_active, self.episode_start)  # [A_t]

        # After first call in an episode, we no longer have a "no_prev" situation
        self.episode_start = False

        # Convert logits to positive Dirichlet concentration parameters
        concentration = F.softplus(logits) + 1e-4    # ensure > 0

        if deterministic:
            # Evaluation: use Dirichlet mean (no sampling, no log_prob)
            self.det_step = False
            w = concentration / concentration.sum()
            return w.detach().cpu().numpy()

        # Training: sample from Dirichlet and use log_prob for REINFORCE
        dist = Dirichlet(concentration)
        weights = dist.sample()                      # [A_t], sum=1, >=0
        log_prob = dist.log_prob(weights)           # scalar

        self.last_logprob = log_prob
        self.det_step = True

        return weights.detach().cpu().numpy()

    # ------------------------------------------------------------------
    # REINFORCE UPDATE HOOK (CALLED PER STEP)
    # ------------------------------------------------------------------
    def update(self, obs, action, reward, next_obs, done):
        """
        Accumulate log_probs and rewards for REINFORCE.
        `reward` is a Python float from the environment (and *should* be detached).
        """
        if self.training and self.det_step:
            # store REINFORCE ingredients: reward scalar + log π(a|s)
            self.rewards_buf.append(float(reward))
            self.logprobs.append(self.last_logprob)
        # return optional metrics (we'll skip for now)
        return None

    # ------------------------------------------------------------------
    # SAVE / LOAD STUBS
    # ------------------------------------------------------------------
    def load(self, path):
        pass

    def save(self, path):
        pass

    # ------------------------------------------------------------------
    # EPISODE HOOKS
    # ------------------------------------------------------------------
    def on_episode_start(self):
        """
        Called by BaseAgent at the beginning of each training episode.
        """
        self.optimizer.zero_grad()
        self.episode_start = True
        self.logprobs = []
        self.rewards_buf = []

    def on_episode_end(self) -> None:
        """
        Called by BaseAgent at the end of each training episode.

        Implements episodic REINFORCE:
          loss = -Σ_t (G_t * log π(a_t | s_t))

        where G_t is the (optionally normalized) return from step t onward.
        """
        # Epsilon decay
        if self.epsilon > self.config.epsilon_end:
            self.epsilon = max(
                self.config.epsilon_end,
                self.epsilon - self.epsilon_decay_per_episode
            )

        if not self.logprobs:
            # Nothing to update (e.g., episode with only random actions)
            return

        # ----- compute returns G_t from rewards -----
        # simple undiscounted MC return
        returns = []
        G = 0.0
        for r in reversed(self.rewards_buf):
            G = r + G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # normalize returns as a cheap baseline
        if returns.numel() > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        log_probs = torch.stack(self.logprobs)

        # REINFORCE loss: negative because we maximize expected return
        loss = -(returns * log_probs).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # ------------------------------------------------------------------
    # EVALUATION (NO GRAD / NO UPDATES)
    # ------------------------------------------------------------------
    def evaluate_on_env(
        self,
        eval_env: PortfolioEnv,
        n_episodes: int = 5,
        deterministic: bool = True,
        max_steps: int = 200,
    ):
        episode_rewards = []

        with torch.no_grad():
            for ep in tqdm(range(n_episodes), leave=False):
                obs = eval_env.reset()
                episode_reward = 0.0
                done = False
                step_count = 0

                # reset flag for no_prev inside evaluation as well
                self.episode_start = True

                while not done and step_count < max_steps:
                    action = self.select_action(obs, deterministic=deterministic)
                    obs, reward, done, info = eval_env.step(action)
                    episode_reward += reward
                    step_count += 1

                episode_rewards.append(episode_reward)

        return {
            'mean_return': float(np.mean(episode_rewards)),
            'std_return': float(np.std(episode_rewards)),
            'min_return': float(np.min(episode_rewards)),
            'max_return': float(np.max(episode_rewards)),
        }
