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
    One GRU over entire market state per timestep.
    Outputs logits for all A assets simultaneously.
    """

    def __init__(self, feature_dim=4, hidden_dim=128, seq_layer = nn.GRU):
        super().__init__()

        A = 10

        self.A = A
        self.input_dim = A * feature_dim

        # One GRU over entire market sequence
        self.gru = seq_layer(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        # MLP head outputs A logits at once
        self.fc1 = nn.Linear(hidden_dim + A + 1, 128)
        self.fc2 = nn.Linear(128, A)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, features, prev_alloc, first_step):
        """
        features:   [A, 60, 4]
        prev_alloc: [A]   (full previous allocation vector)
        first_step: bool
        """

        A = self.A
        device = features.device

        # ---- 1. Reshape to full market sequence ----
        # features: [A, 60, 4] → transpose → [60, A, 4]
        # then reshape to [1, 60, A*4]
        seq = features.permute(1, 0, 2).reshape(1, 60, A * 4)

        # ---- 2. GRU processes the entire market jointly ----
        out, _ = self.gru(seq)              # [1, 60, hidden_dim]
        global_emb = out[:, -1, :]          # [1, hidden_dim]

        # ---- 3. Prepare state additions ----
        prev_alloc_vec = prev_alloc.unsqueeze(0)   # [1, A]

        if first_step:
            prev_alloc_vec = torch.zeros_like(prev_alloc_vec)

        first_step_flag = torch.ones((1, 1), device=device) if first_step else \
                          torch.zeros((1, 1), device=device)

        # ---- 4. Combine into global state ----
        x = torch.cat([global_emb, prev_alloc_vec, first_step_flag], dim=1)

        # ---- 5. MLP head outputs logits for ALL A assets ----
        x = self.act(self.fc1(x))
        logits = self.fc2(x).squeeze(0)     # [A]

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
    """
    Stable Dirichlet Policy Gradient Agent:
      - Dirichlet(alpha) policy over simplex
      - Temperature scales concentration without changing mean
      - REINFORCE update
    """

    def __init__(self, config, env):
        super().__init__(config, env)

        self.config  = config
        self.device  = config.device
        self.max_alloc = config.max_alloc

        # Temperature for Dirichlet
        self.tau = float(getattr(config, "temperature", 1.0))

        # Epsilon for exploration override
        self.epsilon = config.epsilon_start
        self.epsilon_decay_episodes = config.epsilon_decay_episodes
        self.epsilon_decay_per_episode = (
            config.epsilon_start - config.epsilon_end
        ) / self.epsilon_decay_episodes

        # Canonical asset indexer
        self.asset_indexer = AssetIndexer(
            dataset_path=config.dataset_path
        )

        # Policy network → outputs logits → softplus → Dirichlet α
        self.nn = PolicyNet(
            feature_dim=4,
            hidden_dim=config.hidden_dim,
            seq_layer=config.recurrent_layer,
        ).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.nn.parameters(),
            lr=config.learning_rate,
        )

        # Episode bookkeeping
        self.episode_start = True
        self.logprobs = []         # scalar log_probs per step
        self.rewards_buf = []      # python floats
        self.last_logprob = None   # for most recent stochastic action
        self.det_step = False

    def load(self):
      pass
    def save(self):
      pass


    # ======================================================================
    # RANDOM ACTION (for epsilon override)
    # ======================================================================
    def random_policy(self, obs):
        A_t = len(obs["asset_ids"])
        if A_t == 0:
            return np.array([], dtype=np.float32)

        w = np.random.rand(A_t).astype(np.float32)
        w /= w.sum()
        return w


    # ======================================================================
    # ACTION SELECTION
    # ======================================================================
    def select_action(self, obs, deterministic: bool = False):
        """
        deterministic=False:
            - epsilon-greedy allowed
            - stochastic Dirichlet policy
            - REINFORCE log_prob recorded

        deterministic=True:
            - NO epsilon-greedy override
            - STILL stochastic sampling (not eval)
            - REINFORCE log_prob recorded

        (True evaluation should wrap this in torch.no_grad() at a higher level.)
        """

        # --- Reindex canonical assets ---
        features, prev_w, mask = self.asset_indexer.reindex(obs)

        features = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        prev_w   = torch.as_tensor(prev_w, dtype=torch.float32, device=self.device)
        mask     = torch.as_tensor(mask, dtype=torch.bool,    device=self.device)

        features_active = features[mask]     # [A_t, T, F]
        prev_w_active   = prev_w[mask]       # [A_t]
        A_t = prev_w_active.size(0)

        # --- Policy forward pass ---
        logits = self.nn(
            features_active,
            prev_w_active,
            self.episode_start
        )
        self.episode_start = False

        # ==========================================================
        # STEP 1 — EPSILON-GREEDY OVERRIDE
        # ==========================================================
        if (not deterministic) and (self.rng.random() < self.epsilon):
            self.last_logprob = None
            self.det_step = False
            return self.random_policy(obs)

        # ==========================================================
        # STEP 2 — DIRICHLET WITH TEMPERATURE (MEAN-PRESERVING)
        # ==========================================================
        # Convert logits → positive α
        alpha = F.softplus(logits) + 1e-4     # ensures positivity

        # Apply temperature (scales sharpness)
        alpha_scaled = alpha / self.tau

        # 🔥 The CRITICAL FIX:
        # Normalize α so total concentration = A_t (keeps Dirichlet mean unchanged)
        alpha_scaled = alpha_scaled * (A_t / alpha_scaled.sum())

        # Build distribution
        dist = Dirichlet(alpha_scaled)

        # Sample allocation (stochastic even in deterministic=True)
        weights_t = dist.sample()             # [A_t], simplex

        # Compute log_prob of the sampled action (scalar)
        log_prob = dist.log_prob(weights_t)

        # Keep for REINFORCE
        self.last_logprob = log_prob
        self.det_step = False

        # Env receives numpy
        return weights_t.detach().cpu().numpy()


    # ======================================================================
    # STORE REINFORCE DATA
    # ======================================================================
    def update(self, obs, action, reward, next_obs, done):
        """
        Store log_prob and reward for REINFORCE.
        """
        if self.last_logprob is not None:
            self.logprobs.append(self.last_logprob)
            self.rewards_buf.append(float(reward))
        return None


    # ======================================================================
    # EPISODE LIFECYCLE
    # ======================================================================
    def on_episode_start(self):
        self.optimizer.zero_grad()
        self.episode_start = True
        self.logprobs = []
        self.rewards_buf = []
        self.last_logprob = None
        self.det_step = False

    def on_episode_end(self):
        """
        REINFORCE:
            loss = - Σ_t G_t * log π(a_t|s_t)
        """
        # Epsilon decay
        if self.epsilon > self.config.epsilon_end:
            self.epsilon = max(
                self.config.epsilon_end,
                self.epsilon - self.epsilon_decay_per_episode
            )

        if len(self.logprobs) == 0:
            return

        # ---- Compute returns ----
        returns = []
        G = 0.0
        for r in reversed(self.rewards_buf):
            G = r + G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # Normalize returns for variance reduction
        if returns.numel() > 1:
            std = returns.std()
            if std > 1e-6:
                returns = (returns - returns.mean()) / (std + 1e-8)

        log_probs = torch.stack(self.logprobs)  # shape [T]
        assert log_probs.shape == returns.shape

        loss = -(returns * log_probs).sum()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.nn.parameters(), 1.0)
        self.optimizer.step()


    # ======================================================================
    # EVALUATION — TRUE DETERMINISTIC MODE
    # ======================================================================
    def evaluate_on_env(
        self,
        eval_env: PortfolioEnv,
        n_episodes: int = 5,
        deterministic: bool = True,
        max_steps: int = 200,
    ):
        """
        Deterministic evaluation → use *mean* of Dirichlet, not sampling.
        (Handled here, NOT in select_action)
        """
        episode_rewards = []

        with torch.no_grad():
            for ep in range(n_episodes):
                obs = eval_env.reset()
                done = False
                total_reward = 0
                t = 0

                self.episode_start = True

                while not done and t < max_steps:
                    # Rebuild policy input
                    features, prev_w, mask = self.asset_indexer.reindex(obs)
                    features = torch.tensor(features, dtype=torch.float32, device=self.device)
                    prev_w   = torch.tensor(prev_w,   dtype=torch.float32, device=self.device)
                    mask     = torch.tensor(mask,     dtype=torch.bool,    device=self.device)

                    logits = self.nn(
                        features[mask],
                        prev_w[mask],
                        self.episode_start
                    )
                    self.episode_start = False

                    # Deterministic action = Dirichlet mean = alpha / sum(alpha)
                    alpha = F.softplus(logits) + 1e-4
                    w = (alpha / alpha.sum()).cpu().numpy()

                    obs, r, done, info = eval_env.step(w)
                    total_reward += r
                    t += 1

                episode_rewards.append(total_reward)

        return {
            "mean_return": float(np.mean(episode_rewards)),
            "std_return": float(np.std(episode_rewards)),
            "min_return": float(np.min(episode_rewards)),
            "max_return": float(np.max(episode_rewards)),
        }
