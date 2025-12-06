import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet
import numpy as np
from pathlib import Path
import pickle
import json
from data.data_loader import DatabaseConfig
from data.dataset_loader import load_exported_dataset
from data.dataset_backend import DatasetBackend
from environment.environment import PortfolioEnv, EnvConfig

from agents.policy_grad.policygrad import PolicyGradAgent, PolicyGradConfig
from agents.dqn.hyperparameter_search import create_environments

class ReinforceBaselineAgent(PolicyGradAgent):
    """
    REINFORCE with learned value baseline using a shared encoder
    without modifying PolicyNet.
    """

    def __init__(self, config, env):
        super().__init__(config, env)

        # Number of canonical assets
        self.A_canonical = len(self.asset_indexer.canonical_assets)
        self.feature_dim = 4  # You stated features are 4-dim per timestep
        hidden_dim = config.hidden_dim

        # Value head input:
        #   [global_emb(hidden_dim), prev_alloc(A), first_step_flag(1)]
        value_input_dim = hidden_dim + self.A_canonical + 1
        value_hidden_dim = hidden_dim

        self.value_head = nn.Sequential(
            nn.Linear(value_input_dim, value_hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(value_hidden_dim, 1)
        ).to(self.device)

        self.value_coef = float(getattr(config, "value_coef", 0.5))

        # Optimizer must include BOTH policy + value parameters
        self.optimizer = torch.optim.Adam(
            list(self.nn.parameters()) + list(self.value_head.parameters()),
            lr=config.learning_rate
        )

    # ------------------------------------------------------------
    # Reset per-episode buffers
    # ------------------------------------------------------------
    def on_episode_start(self):
        super().on_episode_start()
        self.values = []         # V(s_t) for every timestep
        self.sampled_mask = []   # 1 = policy step, 0 = epsilon step

    # ------------------------------------------------------------
    # Compute baseline V(s_t) using PolicyNet's GRU encoder
    # ------------------------------------------------------------
    def _compute_value(self, features, prev_w, first_step: bool):
        device = features.device
        A = self.A_canonical

        # Build GRU input as PolicyNet does:
        # features: [A, T, F] -> [1, T, A*F]
        seq = features.permute(1, 0, 2).reshape(1, features.size(1), A * self.feature_dim)

        # Shared encoder: directly use PolicyNet's GRU weights
        out, _ = self.nn.gru(seq)            # [1, T, hidden_dim]
        global_emb = out[:, -1, :]           # [1, hidden_dim]

        # full previous allocation (canonical)
        prev_alloc_full = prev_w.unsqueeze(0)       # [1, A]
        if first_step:
            prev_alloc_full = torch.zeros_like(prev_alloc_full)

        fs_flag = torch.ones((1,1), device=device) if first_step else \
                  torch.zeros((1,1), device=device)

        vin = torch.cat([global_emb, prev_alloc_full, fs_flag], dim=1)
        return self.value_head(vin).squeeze(0).squeeze(-1)

    # ------------------------------------------------------------
    # Select action: compute both policy output + baseline V(s)
    # ------------------------------------------------------------
    def select_action(self, obs, deterministic: bool = False):
        # Canonical state
        features, prev_w, mask = self.asset_indexer.reindex(obs)

        features = torch.tensor(features, dtype=torch.float32, device=self.device)
        prev_w   = torch.tensor(prev_w,   dtype=torch.float32, device=self.device)
        mask     = torch.tensor(mask,     dtype=torch.bool,    device=self.device)

        first_step = self.episode_start

        # POLICY FORWARD (full canonical → active subset)
        logits_full = self.nn(features, prev_w, first_step)
        logits = logits_full[mask]

        # Episode now started
        self.episode_start = False

        # ---------- EPSILON-GREEDY ----------
        if (not deterministic) and (self.rng.random() < self.epsilon):
            self.last_logprob = None
            self.last_value   = None
            self.det_step     = False
            return self.random_policy(obs)

        # ---------- DIRICHLET ----------
        alpha = F.softplus(logits) + 1e-4
        alpha_scaled = alpha / self.tau

        A_t = logits.numel()
        alpha_scaled = alpha_scaled * (A_t / alpha_scaled.sum())

        dist = Dirichlet(alpha_scaled)
        w_t = dist.sample()
        log_prob = dist.log_prob(w_t)

        # ---------- BASELINE ----------
        v_t = self._compute_value(features, prev_w, first_step)

        # Store for update
        self.last_logprob = log_prob
        self.last_value   = v_t
        self.det_step = False

        return w_t.detach().cpu().numpy()

    # ------------------------------------------------------------
    # Store timestep information
    # ------------------------------------------------------------
    def update(self, obs, action, reward, next_obs, done):
        """
        Store reward, always store value, store log_prob only if sampled.
        """

        # Always store reward
        self.rewards_buf.append(float(reward))

        # Store baseline V(s_t)
        if self.last_value is not None:
            self.values.append(self.last_value)
        else:
            # epsilon step → compute value retroactively
            features, prev_w, _ = self.asset_indexer.reindex(obs)
            features = torch.tensor(features, dtype=torch.float32, device=self.device)
            prev_w   = torch.tensor(prev_w,   dtype=torch.float32, device=self.device)
            v_t = self._compute_value(features, prev_w, False)
            self.values.append(v_t)

        # log_prob only for sampled steps
        if self.last_logprob is not None:
            self.logprobs.append(self.last_logprob)
            self.sampled_mask.append(1)
        else:
            self.sampled_mask.append(0)

        return None

    def save(self, path):
      super().save(path)
      torch.save(self.value_head.state_dict(), path / "value_weights.pt")

    # ------------------------------------------------------------
    # Episode end: REINFORCE with learned baseline
    # ------------------------------------------------------------
    def on_episode_end(self):
        # epsilon decay
        if self.epsilon > self.config.epsilon_end:
            self.epsilon = max(
                self.config.epsilon_end,
                self.epsilon - self.epsilon_decay_per_episode
            )

        T = len(self.rewards_buf)
        if T == 0:
            return

        # ---------- RETURNS ----------
        returns = []
        G = 0.0
        for r in reversed(self.rewards_buf):
            G = r + G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        values  = torch.stack(self.values)           # [T]
        mask    = torch.tensor(self.sampled_mask, device=self.device)  # [T]

        # ---------- ADVANTAGES ----------
        advantages = returns - values.detach()

        if advantages.numel() > 1:
            std = advantages.std()
            if std > 1e-6:
                advantages = (advantages - advantages.mean()) / (std + 1e-8)

        # ---------- VALUE LOSS ----------
        value_loss = 0.5 * (returns - values).pow(2).sum()

        # ---------- POLICY LOSS (only sampled steps) ----------
        # log_probs matches number of sampled steps
        # ---------- POLICY LOSS (only sampled steps) ----------
        if len(self.logprobs) > 0:
            # log_probs is in SAME ORDER that we appended sampled_mask==1 entries
            log_probs = torch.stack(self.logprobs)              # [num_sampled]

            # Extract only advantages where sampled_mask == 1
            advantages_sampled = advantages[mask.bool()]        # [num_sampled]

            policy_loss = -(advantages_sampled * log_probs).sum()
        else:
            print('No policy action was sampled.')
            # If no policy action was sampled, no policy gradient is applied.
            policy_loss = torch.tensor(0.0, device=self.device)

        # ---------- TOTAL LOSS ----------
        loss = policy_loss + self.value_coef * value_loss

        # ---------- BACKPROP ----------
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.nn.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.value_head.parameters(), 1.0)
        self.optimizer.step()
