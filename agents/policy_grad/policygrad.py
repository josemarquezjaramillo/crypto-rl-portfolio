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
    Cross-asset GRU Policy Network
    --------------------------------
    Inputs:
        features: [A_canonical, T=60, F=4]
        prev_alloc: [A_canonical]
        first_step: bool

    Output:
        logits_full: [A_canonical]
    """

    def __init__(self, A_canonical, feature_dim=4, hidden_dim=128, recurrent_layer=nn.GRU):
        super().__init__()

        self.A = A_canonical
        self.feature_dim = feature_dim
        self.input_dim = A_canonical * feature_dim

        # ---- Global GRU across all assets ----
        self.gru = recurrent_layer(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        # ---- MLP head outputs logits for ALL assets ----
        self.fc1 = nn.Linear(hidden_dim + A_canonical + 1, 128)
        self.fc2 = nn.Linear(128, A_canonical)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, features, prev_alloc, first_step):
        """
        features:   [A, T, F]
        prev_alloc: [A]
        """

        device = features.device
        A = self.A

        # ---- 1. Reshape to a single market sequence ----
        # features: [A, 60, 4] → [60, A, 4] → [1, 60, A*4]
        seq = (
            features.permute(1, 0, 2)
            .reshape(1, features.size(1), A * self.feature_dim)
        )

        # ---- 2. GRU over full market ----
        out, _ = self.gru(seq)          # [1, T, hidden]
        global_emb = out[:, -1, :]      # [1, hidden]

        # ---- 3. Previous full allocation vector ----
        prev_alloc_vec = prev_alloc.unsqueeze(0)  # [1, A]

        if first_step:
            prev_alloc_vec = torch.zeros_like(prev_alloc_vec)

        # ---- 4. First-step flag ----
        fs_flag = torch.ones((1, 1), device=device) if first_step \
                  else torch.zeros((1, 1), device=device)

        # ---- 5. Combine market embedding + prev_alloc + flag ----
        x = torch.cat([global_emb, prev_alloc_vec, fs_flag], dim=1)

        # ---- 6. MLP head produces logits for ALL assets ----
        x = self.act(self.fc1(x))
        logits = self.fc2(x).squeeze(0)          # [A]

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
    Cross-Asset Dirichlet Policy Gradient Agent (REINFORCE)
    """

    def __init__(self, config, env):
        super().__init__(config, env)

        self.config  = config
        self.device  = config.device
        self.max_alloc = config.max_alloc

        # Dirichlet temperature scaling
        self.tau = float(getattr(config, "temperature", 1.0))

        # Epsilon scheduling
        self.epsilon = config.epsilon_start
        self.epsilon_decay_episodes = config.epsilon_decay_episodes
        self.epsilon_decay_per_episode = (
            config.epsilon_start - config.epsilon_end
        ) / self.epsilon_decay_episodes

        # Canonical asset loader
        self.asset_indexer = AssetIndexer(
            dataset_path=config.dataset_path
        )
        A_canonical = len(self.asset_indexer.canonical_assets)

        # ---- POLICY NETWORK ----
        self.nn = PolicyNet(
            A_canonical=A_canonical,
            feature_dim=4,
            hidden_dim=config.hidden_dim,
            recurrent_layer=config.recurrent_layer
        ).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.nn.parameters(),
            lr=config.learning_rate,
        )

        # Episode data
        self.episode_start = True
        self.logprobs = []
        self.rewards_buf = []
        self.last_logprob = None
        self.det_step = False

    def save(self, save_dir: Path):
        """
        Save the Policy Gradient Agent to a directory.
    
        Directory contents:
            policy_weights.pt   - model state_dict
            optimizer.pt        - optimizer state_dict
            agent.pkl           - config + metadata
        """
    
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
        # --------------------------
        # 1. Save policy network
        # --------------------------
        torch.save(
            self.nn.state_dict(),
            save_dir / "policy_weights.pt"
        )
    
        # --------------------------
        # 2. Save optimizer
        # --------------------------
        torch.save(
            self.optimizer.state_dict(),
            save_dir / "optimizer.pt"
        )
    
        # --------------------------
        # 3. Save config + metadata
        # --------------------------
        agent_state = {
            "config": self.config,                     # dataclass (pickleable)
            "agent_name": getattr(self.config, "agent_name", None),
            "epsilon": self.epsilon,
            "episode_start": self.episode_start,
            "rng_state": self.rng.getstate() if hasattr(self, "rng") else None,
        }
    
        with open(save_dir / "agent.pkl", "wb") as f:
            pickle.dump(agent_state, f)
    
        print(f"[PolicyGradAgent] Saved checkpoint to: {save_dir}")

    def load(self, load_dir: Path):
        """
        Load the Policy Gradient Agent from a directory.
        """
        load_dir = Path(load_dir)
    
        # --------------------------
        # 1. Load policy weights
        # --------------------------
        policy_path = load_dir / "policy_weights.pt"
        if policy_path.exists():
            state = torch.load(policy_path, map_location=self.device)
            self.nn.load_state_dict(state)
        else:
            raise FileNotFoundError(f"Missing {policy_path}")
    
        # --------------------------
        # 2. Load optimizer
        # --------------------------
        optim_path = load_dir / "optimizer.pt"
        if optim_path.exists():
            opt_state = torch.load(optim_path, map_location=self.device)
            self.optimizer.load_state_dict(opt_state)
        else:
            print("[PolicyGradAgent] Warning: optimizer.pt not found (OK for inference).")
    
        # --------------------------
        # 3. Load metadata
        # --------------------------
        pkl_path = load_dir / "agent.pkl"
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                agent_state = pickle.load(f)
    
            # restore config
            if "config" in agent_state and agent_state["config"] is not None:
                self.config = agent_state["config"]
    
            # restore name
            if "agent_name" in agent_state:
                self.config.agent_name = agent_state["agent_name"]
    
            # restore training state
            self.epsilon = agent_state.get("epsilon", self.epsilon)
            self.episode_start = agent_state.get("episode_start", True)
    
            # RNG restore if available
            if agent_state.get("rng_state") is not None and hasattr(self, "rng"):
                try:
                    self.rng.setstate(agent_state["rng_state"])
                except Exception:
                    print("[PolicyGradAgent] Warning: RNG state incompatible, skipping.")
    
        else:
            print("[PolicyGradAgent] Warning: agent.pkl not found (config not restored).")
    
        print(f"[PolicyGradAgent] Loaded checkpoint from: {load_dir}")



    # =========================================================
    # RANDOM ACTION
    # =========================================================
    def random_policy(self, obs):
        A_t = len(obs["asset_ids"])
        if A_t == 0:
            return np.array([], dtype=np.float32)
        w = np.random.rand(A_t).astype(np.float32)
        return w / w.sum()


    # =========================================================
    # ACTION SELECTION
    # =========================================================
    def select_action(self, obs, deterministic=False):
        """
        Masking is applied AFTER forward pass → critical fix.
        """

        # ---- Canonical reindex ----
        features, prev_w, mask = self.asset_indexer.reindex(obs)

        features = torch.tensor(features, dtype=torch.float32, device=self.device)
        prev_w   = torch.tensor(prev_w,   dtype=torch.float32, device=self.device)
        mask     = torch.tensor(mask,     dtype=torch.bool,    device=self.device)

        # ---- Policy forward on FULL canonical set ----
        logits_full = self.nn(features, prev_w, self.episode_start)
        logits = logits_full[mask]            # select active assets
        prev_w_active = prev_w[mask]

        self.episode_start = False

        # ---- EPSILON OVERRIDE ----
        if (not deterministic) and (self.rng.random() < self.epsilon):
            self.last_logprob = None
            self.det_step = False
            return self.random_policy(obs)

        # ---- DIRICHLET SAMPLING ----
        alpha = F.softplus(logits) + 1e-4
        alpha_scaled = alpha / self.tau

        # mean-preserving normalization
        A_t = logits.numel()
        alpha_scaled = alpha_scaled * (A_t / alpha_scaled.sum())

        dist = Dirichlet(alpha_scaled)
        w_t = dist.sample()
        log_prob = dist.log_prob(w_t)

        # Store log_prob for REINFORCE
        self.last_logprob = log_prob
        self.det_step = False

        return w_t.detach().cpu().numpy()


    # =========================================================
    # STORE STEP INFO
    # =========================================================
    def update(self, obs, action, reward, next_obs, done):
        if self.last_logprob is not None:
            self.logprobs.append(self.last_logprob)
            self.rewards_buf.append(float(reward))
        return None


    # =========================================================
    # EPISODE LIFECYCLE
    # =========================================================
    def on_episode_start(self):
        self.optimizer.zero_grad()
        self.episode_start = True
        self.logprobs = []
        self.rewards_buf = []
        self.last_logprob = None
        self.det_step = False

    def on_episode_end(self):
        if self.epsilon > self.config.epsilon_end:
            self.epsilon = max(
                self.config.epsilon_end,
                self.epsilon - self.epsilon_decay_per_episode
            )

        if len(self.logprobs) == 0:
            return

        # ---- COMPUTE RETURNS ----
        returns = []
        G = 0.0
        for r in reversed(self.rewards_buf):
            G = r + G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        if returns.numel() > 1:
            std = returns.std()
            if std > 1e-6:
                returns = (returns - returns.mean()) / (std + 1e-8)

        log_probs = torch.stack(self.logprobs)
        assert log_probs.shape == returns.shape

        loss = -(returns * log_probs).sum()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.nn.parameters(), 1.0)
        self.optimizer.step()


    # =========================================================
    # EVALUATION MODE
    # =========================================================
    @torch.no_grad()
    def evaluate_on_env(self, eval_env, n_episodes=5, deterministic=True, max_steps=200):
        episode_rewards = []

        for ep in range(n_episodes):
            obs = eval_env.reset()
            done = False
            total_reward = 0
            t = 0
            self.episode_start = True

            while not done and t < max_steps:

                features, prev_w, mask = self.asset_indexer.reindex(obs)
                features = torch.tensor(features, dtype=torch.float32, device=self.device)
                prev_w   = torch.tensor(prev_w,   dtype=torch.float32, device=self.device)
                mask     = torch.tensor(mask,     dtype=torch.bool,    device=self.device)

                logits_full = self.nn(features, prev_w, self.episode_start)
                logits = logits_full[mask]
                self.episode_start = False

                alpha = F.softplus(logits) + 1e-4
                w = (alpha / alpha.sum()).cpu().numpy()

                obs, r, done, info = eval_env.step(w)
                total_reward += r
                t += 1

            episode_rewards.append(total_reward)

        return {
            "mean_return": float(np.mean(episode_rewards)),
            "std_return":  float(np.std(episode_rewards)),
            "min_return":  float(np.min(episode_rewards)),
            "max_return":  float(np.max(episode_rewards)),
        }
