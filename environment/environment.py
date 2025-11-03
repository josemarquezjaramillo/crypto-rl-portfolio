"""
Daily-rebalancing portfolio environment for cryptocurrency RL.

This module implements a Markov Decision Process (MDP) for portfolio management
with daily rebalancing, transaction costs, and realistic trading constraints as
specified in DATA_METHODS.md.

The environment enforces:
- Long-only, fully invested portfolios (no cash position)
- Daily rebalancing with configurable transaction costs
- Turnover caps to model liquidity constraints
- Per-asset concentration limits
- Monthly universe freeze with cold-start eligibility rules

Key Components
--------------
- EnvConfig: Frozen configuration dataclass for reproducible experiments
- DataBackend: Abstract interface for leakage-safe data access (implemented by
  data.dataset_backend.DatasetBackend)
- PortfolioEnv: Main environment class implementing the MDP
- Obs: Observation TypedDict returned to agents
- StepInfo: Diagnostic information returned on each step

Data Flow
---------
The environment uses precomputed dataset artifacts exported by data_exporter.py
and loaded via dataset_loader.py. A DatasetBackend adapter bridges the data layer
to the environment, ensuring no look-ahead leakage.

    Raw OHLCV → data_builder → data_exporter → dataset_v1/
                                                    ↓
    dataset_loader → ExportedDataset → DatasetBackend → PortfolioEnv

Usage Example (Continuous Actions)
----------------------------------
>>> from pathlib import Path
>>> from data.dataset_loader import load_exported_dataset
>>> from data.dataset_backend import DatasetBackend
>>> from environment.environment import PortfolioEnv, EnvConfig
>>>
>>> # Load dev split and filter for training
>>> ds = load_exported_dataset("dataset_v1", split="dev")
>>> train_backend = DatasetBackend(ds, split_tag_filter="train_core")
>>>
>>> # Configure environment
>>> cfg = EnvConfig(
>>>     split="train",
>>>     cost_rate=0.001,      # 10 bps transaction cost
>>>     turnover_cap=0.30,    # 30% daily turnover limit
>>>     action_mode="continuous",
>>>     random_seed=42,
>>> )
>>>
>>> # Create and run environment
>>> env = PortfolioEnv(cfg, train_backend)
>>> obs = env.reset(seed=42)
>>> 
>>> for step in range(100):
>>>     action = env.sample_action()  # or use your policy
>>>     obs, reward, done, info = env.step(action)
>>>     
>>>     if step % 10 == 0:
>>>         print(f"Step {step}: PV={info['portfolio_value']:.4f}, "
>>>               f"Reward={reward:.6f}")
>>>     
>>>     if done:
>>>         break

Usage Example (Validation Split)
--------------------------------
>>> # Use same dataset but filter for validation window
>>> val_backend = DatasetBackend(ds, split_tag_filter="val_window_val_bear")
>>> env_val = PortfolioEnv(cfg, val_backend)
>>> obs = env_val.reset()
>>> # Evaluate policy without training...

Usage Example (With CSV Logging)
--------------------------------
>>> cfg_logged = EnvConfig(
>>>     split="train",
>>>     cost_rate=0.001,
>>>     turnover_cap=0.30,
>>>     log_dir=Path("logs/experiment_001"),  # Enable CSV logging
>>>     random_seed=42,
>>> )
>>> env = PortfolioEnv(cfg_logged, train_backend)
>>> obs = env.reset()
>>> # Run episode...
>>> env.close()  # Flush logs

References
----------
See data/DATA_METHODS.md for detailed specifications on:
- Market data and investable universe
- State representation and normalization
- Action space and constraints
- Reward computation
- Dataset export format

Citations
---------
- Jiang et al. (2017): "A Deep Reinforcement Learning Framework for the
  Financial Portfolio Management Problem" (EIIE architecture)
- Duchi et al. (2008): "Efficient Projections onto the l1-Ball for Learning
  in High Dimensions" (simplex projection algorithm)
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TypedDict, Literal, Optional, Tuple, Dict, List, Any, Union

import numpy as np
import numpy.typing as npt


# ---------- Observation & Info types ----------

class Obs(TypedDict, total=False):
    """
    Observation returned to the agent at each decision time.
    
    Fields
    ------
    features : np.ndarray
        Shape [A_t, 4, 60], dtype float32.
        Per-asset OHLCV lookback tensor. Channels: [Close, High, Low, Volume].
        Normalized per DATA_METHODS.md specification.
    prev_weights : np.ndarray
        Shape [A_t], dtype float32.
        Portfolio weights realized at t-1 (before current decision).
        Aligned to asset_ids order.
    asset_ids : List[str]
        Length A_t. Asset identifiers (row labels for features/weights).
    date : np.datetime64
        Decision date t (day precision, timezone-naive).
    action_mask : np.ndarray, optional
        Shape [A_max], dtype bool. Only present if cfg.return_action_mask=True.
        mask[i] = True for valid assets (i < A_t), False for padding (i >= A_t).
        Used for batching environments with variable universe sizes.
    """
    features: npt.NDArray[np.float32]      # shape: [A_t, 4, 60]
    prev_weights: npt.NDArray[np.float32]  # shape: [A_t]
    asset_ids: List[str]                   # row labels
    date: np.datetime64                    # decision date t
    action_mask: npt.NDArray[np.bool_]     # shape: [A_max], optional


class StepInfo(TypedDict, total=False):
    """Diagnostics for training/eval (returned on every step)."""
    date: np.datetime64
    tradable_assets: List[str]
    proposed_weights: Optional[npt.NDArray[np.float32]]
    executed_weights: npt.NDArray[np.float32]
    discrete_action_idx: Optional[int]
    gross_log_return: float               # log(w^T * (1 + r_{t+1}))
    turnover: float                       # L1 change ||w_t - w_{t-1}||_1
    transaction_cost: float               # c * turnover
    reward_net: float                     # gross - cost
    universe_event: Optional[str]         # "monthly_roll","entry","exit", or None
    portfolio_value: float
    constraints_active: Dict[str, bool]   # {"simplex":..., "nonneg":..., "cap":..., "turnover":...}


# ---------- Config ----------

@dataclass(frozen=True)
class EnvConfig:
    """
    Frozen configuration for reproducible environment behavior.
    
    All parameters are immutable after construction to ensure experiment
    reproducibility. Create new instances for different configurations.
    
    Parameters
    ----------
    split : {"train", "val", "test", "dev"}
        Split identifier for logging/metadata. Does not affect backend filtering
        (use DatasetBackend's split_tag_filter for that).
    lookback_days : int, default=60
        Number of calendar days in the observation window. Must match the
        dataset export configuration.
    rebalance_interval_days : int, default=1
        Portfolio rebalancing frequency in days. Currently only daily (1) is
        supported per DATA_METHODS.md.
    cost_rate : float, default=0.001
        Proportional transaction cost rate (e.g., 0.001 = 10 basis points).
        Applied to L1 turnover: cost = cost_rate * ||w_t - w_{t-1}||_1.
    turnover_cap : float | None, default=0.30
        Daily turnover limit (L1 norm). Set to None to disable. Value of 0.30
        means max 30% of portfolio can be rebalanced per day.
    max_weight_per_asset : float | None, default=None
        Per-asset concentration limit (e.g., 0.25 = 25% max per asset).
        Set to None to disable.
    action_mode : {"continuous", "discrete"}
        - "continuous": agent outputs target weights directly (e.g., policy gradient)
        - "discrete": agent outputs index into discrete_action_map (e.g., DQN)
    discrete_action_map : np.ndarray | None, default=None
        Shape [K, A_max]. Required if action_mode="discrete".
        Each row is either absolute target weights or deltas to apply to prev_w.
    random_seed : int, default=42
        Seed for environment-level randomness (action sampling, tie-breaking).
    strict_projection : bool, default=True
        Always project actions to feasible set. Should remain True for fair
        comparisons across agents.
    log_dir : Path | None, default=None
        Directory for CSV logs. If set, environment writes per-step metrics
        to {log_dir}/env_{seed}_{timestamp}.csv. Call env.close() to flush.
    start_date : str | None, default=None
        ISO date (YYYY-MM-DD) to constrain episode start. Filters backend dates.
    end_date : str | None, default=None
        ISO date (YYYY-MM-DD) to constrain episode end. Filters backend dates.
    return_action_mask : bool, default=False
        If True, observations include 'action_mask' field for batching support.
        Requires action_mask_size to be set.
    action_mask_size : int | None, default=None
        Maximum universe size (A_max) for action mask padding. Required if
        return_action_mask=True.
    
    Examples
    --------
    Basic training configuration:
    
    >>> cfg = EnvConfig(
    ...     split="train",
    ...     cost_rate=0.0010,
    ...     turnover_cap=0.30,
    ...     random_seed=42
    ... )
    
    Configuration with logging enabled:
    
    >>> cfg = EnvConfig(
    ...     split="train",
    ...     cost_rate=0.0010,
    ...     turnover_cap=0.30,
    ...     log_dir=Path("logs/exp001"),
    ...     random_seed=42
    ... )
    
    Configuration for batched training with variable universe sizes:
    
    >>> cfg = EnvConfig(
    ...     split="train",
    ...     cost_rate=0.0010,
    ...     return_action_mask=True,
    ...     action_mask_size=50,  # pad all obs to 50 assets
    ...     random_seed=42
    ... )
    """
    split: Literal["train", "val", "test", "dev"] = "train"
    lookback_days: int = 60
    rebalance_interval_days: int = 1
    cost_rate: float = 0.0010
    turnover_cap: Optional[float] = 0.30
    max_weight_per_asset: Optional[float] = None
    action_mode: Literal["continuous", "discrete"] = "continuous"
    discrete_action_map: Optional[npt.NDArray[np.float32]] = None
    random_seed: int = 42
    strict_projection: bool = True
    log_dir: Optional[Path] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    return_action_mask: bool = False
    action_mask_size: Optional[int] = None


# ---------- Data backend interface ----------

class DataBackend:
    """
    Abstract interface for leakage-safe, precomputed dataset access.
    
    Implementations must provide access to normalized feature tensors and
    forward returns without exposing future information at decision time.
    
    The canonical implementation is DatasetBackend in data/dataset_backend.py,
    which wraps ExportedDataset from data/dataset_loader.py.
    
    Contract
    --------
    - dates() must return chronologically sorted decision dates
    - get_day(date) must return data aligned to date (no look-ahead)
    - Forward returns are for reward computation only (not in observation)
    - All tensors must be float32 for consistency
    
    Examples
    --------
    Using the concrete DatasetBackend implementation:
    
    >>> from data.dataset_loader import load_exported_dataset
    >>> from data.dataset_backend import DatasetBackend
    >>> 
    >>> ds = load_exported_dataset("dataset_v1", split="dev")
    >>> backend = DatasetBackend(ds, split_tag_filter="train_core")
    >>> 
    >>> dates = backend.dates()  # np.ndarray[np.datetime64]
    >>> features, asset_ids, fwd_r = backend.get_day(dates[0])
    
    See Also
    --------
    data.dataset_backend.DatasetBackend : Concrete implementation
    data.dataset_loader.ExportedDataset : Underlying data container
    """

    def dates(self) -> npt.NDArray[np.datetime64]:
        """
        Return ordered decision dates for the split.
        
        Returns
        -------
        np.ndarray
            Shape (N,), dtype datetime64[D]. Chronologically sorted.
        """
        raise NotImplementedError

    def get_day(self, date: np.datetime64) -> Tuple[
        npt.NDArray[np.float32],  # features: [A_t, 4, 60], normalized per spec
        List[str],                # asset_ids aligned to rows
        npt.NDArray[np.float32],  # fwd_returns: [A_t] simple returns from t->t+1
    ]:
        """
        Fetch precomputed data for a specific decision date.
        
        Parameters
        ----------
        date : np.datetime64
            Decision date t to fetch.
        
        Returns
        -------
        features : np.ndarray
            Shape [A_t, 4, 60], dtype float32. Normalized OHLCV lookback.
        asset_ids : List[str]
            Length A_t. Asset identifiers aligned to feature rows.
        fwd_returns : np.ndarray
            Shape [A_t], dtype float32. Simple returns from t → t+1.
            Used for reward computation; NOT part of agent observation.
        """
        raise NotImplementedError


# ---------- Environment ----------

class PortfolioEnv:
    """
    Daily-rebalancing cryptocurrency portfolio management environment.
    
    This class implements a Markov Decision Process for portfolio optimization
    with realistic constraints, transaction costs, and monthly universe dynamics.
    Designed for comparing policy-gradient, DQN, and bandit-based RL agents under
    identical conditions per DATA_METHODS.md.
    
    Environment Dynamics
    -------------------
    - **State**: Features [A_t, 4, 60] + prev_weights [A_t] + asset_ids + date
    - **Action**: Target portfolio weights (continuous) or catalog index (discrete)
    - **Reward**: Gross log return minus transaction costs
    - **Constraints**: Long-only, fully invested, turnover cap, optional concentration limits
    - **Episode**: Sequence of daily decisions from start_date to end_date
    
    Constraint Enforcement
    ---------------------
    Actions are projected to the feasible set in this order:
    1. Non-negativity (clip to 0)
    2. Simplex (normalize to sum=1)
    3. Per-asset cap (if configured)
    4. Turnover cap (convex blend toward prev_w if needed)
    
    Agents are trained on the *executed* weights after projection, ensuring
    realistic behavior under trading constraints.
    
    Universe Dynamics
    ----------------
    - **Monthly freeze**: Tradable assets fixed for each calendar month
    - **Cold start**: Assets need 60 consecutive clean days before entry
    - **Forced exits**: Assets dropping out are liquidated at t, weights redistributed
    - **New entrants**: Start with 0 weight, can be allocated to at next decision
    
    Data Integration
    ---------------
    The environment accesses precomputed dataset artifacts via a DataBackend
    adapter (typically data.dataset_backend.DatasetBackend), which wraps
    ExportedDataset from data.dataset_loader. This ensures:
    - No look-ahead leakage (features lagged, forward returns hidden from agent)
    - Reproducible splits (train/val/test via split_tag filtering)
    - Efficient access (tensors precomputed and memory-mapped)
    
    Parameters
    ----------
    cfg : EnvConfig
        Environment configuration (frozen dataclass).
    data_backend : DataBackend
        Data adapter implementing dates() and get_day() interface.
        Typically DatasetBackend(ExportedDataset, split_tag_filter="train_core").
    
    Attributes
    ----------
    cfg : EnvConfig
        Configuration reference.
    ds : DataBackend
        Backend reference for data access.
    portfolio_value : float
        Cumulative portfolio value (starts at 1.0).
    cum_turnover : float
        Cumulative L1 turnover across episode.
    cum_costs : float
        Cumulative transaction costs across episode.
    
    Methods
    -------
    reset(seed=None)
        Reset to first date, return initial observation.
    step(action)
        Execute action, advance one day, return (obs, reward, done, info).
    get_state()
        Return current observation dict.
    get_terminal_flag()
        Check if episode has ended.
    seed(seed)
        Set RNG seed for reproducibility.
    sample_action()
        Sample random feasible action (for testing).
    render(mode="text")
        Print current state to console.
    close()
        Flush and close log file (if logging enabled).
    
    Examples
    --------
    Basic usage with training data:
    
    >>> from data.dataset_loader import load_exported_dataset
    >>> from data.dataset_backend import DatasetBackend
    >>> from environment.environment import PortfolioEnv, EnvConfig
    >>> 
    >>> # Load and prepare data
    >>> ds = load_exported_dataset("dataset_v1", split="dev")
    >>> backend = DatasetBackend(ds, split_tag_filter="train_core")
    >>> 
    >>> # Create environment
    >>> cfg = EnvConfig(split="train", cost_rate=0.001, turnover_cap=0.30)
    >>> env = PortfolioEnv(cfg, backend)
    >>> 
    >>> # Run episode
    >>> obs = env.reset(seed=42)
    >>> total_reward = 0.0
    >>> 
    >>> for step in range(100):
    >>>     action = env.sample_action()  # replace with policy
    >>>     obs, reward, done, info = env.step(action)
    >>>     total_reward += reward
    >>>     
    >>>     if done:
    >>>         break
    >>> 
    >>> print(f"Final PV: {env.portfolio_value:.4f}")
    >>> print(f"Total reward: {total_reward:.6f}")
    
    Validation evaluation:
    
    >>> # Use different backend for validation
    >>> val_backend = DatasetBackend(ds, split_tag_filter="val_window_val_bear")
    >>> env_val = PortfolioEnv(cfg, val_backend)
    >>> 
    >>> obs = env_val.reset(seed=123)
    >>> # Evaluate policy in validation regime...
    
    With CSV logging:
    
    >>> cfg_logged = EnvConfig(
    >>>     split="train",
    >>>     cost_rate=0.001,
    >>>     log_dir=Path("logs/exp001"),
    >>>     random_seed=42
    >>> )
    >>> env = PortfolioEnv(cfg_logged, backend)
    >>> obs = env.reset()
    >>> # ... run episode ...
    >>> env.close()  # Flush logs to disk
    
    Notes
    -----
    - Forward returns are used internally for reward computation but are never
      exposed to the agent (no look-ahead leakage).
    - The environment always operates on precomputed, frozen dataset artifacts.
    - Transaction costs are charged on L1 turnover: cost = c * ||w_t - w_{t-1}||_1.
    - Monthly universe changes are handled automatically via align_weights().
    
    See Also
    --------
    EnvConfig : Configuration dataclass
    DataBackend : Abstract data interface
    data.dataset_backend.DatasetBackend : Concrete backend implementation
    """

    def __init__(self, cfg: EnvConfig, data_backend: DataBackend) -> None:
        self.cfg = cfg
        self.ds = data_backend
        self.rng = np.random.default_rng(cfg.random_seed)

        # Validate action mask configuration
        if cfg.return_action_mask and cfg.action_mask_size is None:
            raise ValueError(
                "action_mask_size must be set when return_action_mask=True"
            )

        # Runtime state
        self._dates: npt.NDArray[np.datetime64] = self.ds.dates()
        if cfg.start_date is not None:
            self._dates = self._dates[self._dates >= np.datetime64(cfg.start_date)]
        if cfg.end_date is not None:
            self._dates = self._dates[self._dates <= np.datetime64(cfg.end_date)]
        if len(self._dates) < 2:
            raise ValueError("Insufficient dates after applying start/end constraints.")

        self.t_i: int = 0
        self.date_t: np.datetime64 = self._dates[0]
        self.asset_ids_t: List[str] = []
        self.X_t: npt.NDArray[np.float32] = np.empty((0, 4, self.cfg.lookback_days), dtype=np.float32)
        self.fwd_r_next: npt.NDArray[np.float32] = np.empty((0,), dtype=np.float32)
        self.prev_w: npt.NDArray[np.float32] = np.empty((0,), dtype=np.float32)

        self.portfolio_value: float = 1.0
        self.episode_done: bool = False

        # Cumulative logs
        self.cum_turnover: float = 0.0
        self.cum_costs: float = 0.0
        self._step_counter: int = 0

        # CSV logging
        self._log_file: Optional[Any] = None
        self._log_writer: Optional[Any] = None
        if cfg.log_dir is not None:
            self._init_logger()

    # ----- Public API -----

    def reset(self, seed: Optional[int] = None) -> Obs:
        """
        Reset to the first actionable day in the (possibly constrained) split.
        Initializes previous weights to equal-weight over tradable names.
        Returns the first observation.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.t_i = 0
        self.episode_done = False
        self.portfolio_value = 1.0
        self.cum_turnover = 0.0
        self.cum_costs = 0.0
        self._step_counter = 0

        self._load_day_state(self.t_i)
        A = len(self.asset_ids_t)
        self.prev_w = np.full(A, 1.0 / max(A, 1), dtype=np.float32) if A > 0 else np.array([], dtype=np.float32)

        return self.get_state()

    def get_state(self) -> Obs:
        """
        Return the current observation (features, prev_weights, asset_ids, date).
        
        If cfg.return_action_mask=True, also includes an action mask for batching.
        """
        obs = Obs(
            features=self.X_t,
            prev_weights=self.prev_w.copy(),
            asset_ids=list(self.asset_ids_t),
            date=self.date_t,
        )
        
        # Add action mask if requested (for batching with variable A_t)
        if self.cfg.return_action_mask:
            A_t = len(self.asset_ids_t)
            A_max = self.cfg.action_mask_size
            assert A_max is not None, "action_mask_size must be set"
            
            mask = np.zeros(A_max, dtype=np.bool_)
            mask[:A_t] = True
            obs['action_mask'] = mask
        
        return obs

    def execute_action(self, action: npt.NDArray[np.float32] | int) -> Tuple[Obs, float, bool, StepInfo]:
        """
        Apply action at date t, project to feasible set, compute net reward using
        precomputed forward returns for t->t+1, advance one day, and return:
        (next_obs, reward, done, info).
        """
        self._assert_not_done()

        # Map discrete -> proposal or accept continuous proposal
        discrete_idx: Optional[int] = None
        if self.cfg.action_mode == "discrete":
            if not isinstance(action, (int, np.integer)):
                raise TypeError("Discrete mode expects an integer action index.")
            discrete_idx = int(action)
            w_prop = self._map_discrete_action(discrete_idx)
        else:
            if not isinstance(action, np.ndarray):
                raise TypeError("Continuous mode expects a numpy array of target weights.")
            w_prop = action.astype(np.float32, copy=False)

        # Project to feasible & compute turnover
        w_exec, constraints = self.project_to_feasible(w_prop, self.prev_w)

        # Compute one-step gross log return and costs
        gross_log = self.compute_gross_log_return(w_exec, self.fwd_r_next)
        turnover = float(np.abs(w_exec - self.prev_w).sum())
        cost = float(self.cfg.cost_rate * turnover)
        reward = gross_log - cost

        # Update wealth & logs
        self.portfolio_value *= float(np.exp(gross_log - cost))
        self.cum_turnover += turnover
        self.cum_costs += cost
        self._step_counter += 1

        # Advance one day; remap weights to next day's universe
        old_assets = self.asset_ids_t
        self._advance_one_day()
        self.prev_w = self.align_weights(w_exec, old_assets, self.asset_ids_t)

        info: StepInfo = StepInfo(
            date=self.date_t,
            tradable_assets=list(self.asset_ids_t),
            proposed_weights=w_prop if self.cfg.action_mode == "continuous" else None,
            executed_weights=w_exec,
            discrete_action_idx=discrete_idx,
            gross_log_return=float(gross_log),
            turnover=turnover,
            transaction_cost=cost,
            reward_net=float(reward),
            universe_event=None,  # filled if your backend flags month roll/entries/exits
            portfolio_value=float(self.portfolio_value),
            constraints_active=constraints,
        )

        # Log step metrics if logging enabled
        if self.cfg.log_dir is not None:
            self._log_step(info, action)

        done = self.get_terminal_flag()
        obs_next = self.get_state() if not done else self._terminal_state_like()
        return obs_next, float(reward), bool(done), info

    # Alias to match common RL APIs
    step = execute_action

    def get_terminal_flag(self) -> bool:
        """True when we cannot form a valid forward return for t->t+1 (last usable index)."""
        return self.episode_done or (self.t_i >= len(self._dates) - 1)

    def seed(self, seed: int) -> None:
        """Set RNG for deterministic env-level randomness (tie-breaks, sampling)."""
        self.rng = np.random.default_rng(seed)

    def sample_action(self) -> npt.NDArray[np.float32] | int:
        """Uniform sample from feasible set (for smoke tests)."""
        if self.cfg.action_mode == "discrete":
            if self.cfg.discrete_action_map is None:
                raise ValueError("No discrete_action_map configured.")
            K = self.cfg.discrete_action_map.shape[0]
            return int(self.rng.integers(0, K))
        # Continuous: Dirichlet then project
        A = len(self.asset_ids_t)
        if A == 0:
            return np.array([], dtype=np.float32)
        prop = self.rng.dirichlet(alpha=np.ones(A, dtype=np.float32)).astype(np.float32)
        w_exec, _ = self.project_to_feasible(prop, self.prev_w)
        return w_exec

    def render(self, mode: Literal["text", "none"] = "text") -> None:
        if mode != "text":
            return
        print(f"[{str(self.date_t)}] PV={self.portfolio_value:.6f}  A={len(self.asset_ids_t)}")

    def close(self) -> None:
        """
        Flush and close log file.
        
        Call this at the end of an episode to ensure all log entries are written
        to disk. Safe to call multiple times or if logging is disabled.
        
        Examples
        --------
        >>> env = PortfolioEnv(cfg, backend)
        >>> obs = env.reset()
        >>> # ... run episode ...
        >>> env.close()
        """
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
            self._log_writer = None

    # ----- CSV Logging -----

    def _init_logger(self) -> None:
        """
        Initialize CSV logger with unique filename.
        
        Creates log directory if needed and opens a CSV file for writing
        per-step metrics.
        """
        assert self.cfg.log_dir is not None
        
        # Create directory if needed
        log_dir = Path(self.cfg.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename: env_{split}_{seed}_{timestamp}.csv
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"env_{self.cfg.split}_{self.cfg.random_seed}_{timestamp}.csv"
        log_path = log_dir / filename
        
        # Open file and create CSV writer
        self._log_file = open(log_path, 'w', newline='')
        self._log_writer = csv.writer(self._log_file)
        
        # Write header
        header = [
            'step',
            'date',
            'n_assets',
            'turnover',
            'transaction_cost',
            'gross_log_return',
            'reward_net',
            'portfolio_value',
            'constraint_nonneg',
            'constraint_simplex',
            'constraint_cap',
            'constraint_turnover',
        ]
        self._log_writer.writerow(header)
        self._log_file.flush()

    def _log_step(self, info: StepInfo, action: Union[npt.NDArray[np.float32], int]) -> None:
        """
        Log step metrics to CSV.
        
        Parameters
        ----------
        info : StepInfo
            Step information dict from execute_action().
        action : np.ndarray | int
            Action taken (for potential future logging).
        """
        if self._log_writer is None:
            return
        
        row = [
            self._step_counter,
            str(info['date']),
            len(info['tradable_assets']),
            f"{info['turnover']:.6f}",
            f"{info['transaction_cost']:.6f}",
            f"{info['gross_log_return']:.6f}",
            f"{info['reward_net']:.6f}",
            f"{info['portfolio_value']:.6f}",
            int(info['constraints_active']['nonneg']),
            int(info['constraints_active']['simplex']),
            int(info['constraints_active']['cap']),
            int(info['constraints_active']['turnover']),
        ]
        self._log_writer.writerow(row)
        
        # Flush every 10 steps for efficiency
        if self._step_counter % 10 == 0:
            self._log_file.flush()

    # ----- Internal helpers -----

    def _assert_not_done(self) -> None:
        if self.get_terminal_flag():
            raise RuntimeError("Episode has terminated. Call reset().")

    def _load_day_state(self, idx: int) -> None:
        """Load tensors for the decision date at index `idx`."""
        self.t_i = int(idx)
        self.date_t = self._dates[self.t_i]
        X, asset_ids, fwd_r = self.ds.get_day(self.date_t)
        self.X_t = X.astype(np.float32, copy=False)
        self.asset_ids_t = asset_ids
        self.fwd_r_next = fwd_r.astype(np.float32, copy=False)

    def _advance_one_day(self) -> None:
        """Move from t to t+1 and load next-day tensors. Set done if at the end."""
        if self.t_i >= len(self._dates) - 1:
            self.episode_done = True
            return
        self._load_day_state(self.t_i + 1)
        if self.t_i >= len(self._dates) - 1:
            self.episode_done = True

    def _map_discrete_action(self, idx: int) -> npt.NDArray[np.float32]:
        """
        Map a discrete index to a proposal. Supports either:
          - absolute target weights (simplex rows), or
          - signed deltas to be applied to prev_w before projection.
        Rows must align to current asset_ids order.
        """
        M = self.cfg.discrete_action_map
        if M is None:
            raise ValueError("discrete_action_map is None in discrete mode.")
        if idx < 0 or idx >= M.shape[0]:
            raise IndexError(f"Discrete action idx {idx} out of range [0,{M.shape[0]-1}].")

        row = M[idx].astype(np.float32, copy=False)
        # Heuristic: if any negative present, treat as delta; else absolute target
        if (row < 0).any():
            prop = self.prev_w + row
        else:
            prop = row
        # Truncate/expand to match current A_t if map is built at A_max
        A = len(self.asset_ids_t)
        if prop.shape[0] != A:
            prop = prop[:A] if prop.shape[0] > A else np.pad(prop, (0, A - prop.shape[0]))
        return prop

    def _terminal_state_like(self) -> Obs:
        """Return a terminal observation-like payload for API consistency."""
        return Obs(features=self.X_t, prev_weights=self.prev_w, asset_ids=list(self.asset_ids_t), date=self.date_t)

    # ----- Math utilities (unit-testable) -----

    @staticmethod
    def simplex_projection(v: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Project v onto the probability simplex {w >= 0, sum w = 1}.
        
        Uses the efficient algorithm from Duchi et al. (2008): "Efficient
        Projections onto the l1-Ball for Learning in High Dimensions"
        
        The algorithm:
        1. Sort values in descending order
        2. Find threshold rho such that elements above rho sum close to 1
        3. Shift all elements by threshold theta and clip to [0, inf)
        4. Renormalize for numerical stability
        
        Complexity: O(n log n) due to sorting.
        
        Parameters
        ----------
        v : np.ndarray
            Input vector (any finite values).
        
        Returns
        -------
        np.ndarray
            Projection onto simplex. Same shape as input, dtype float32.
            Satisfies: w >= 0, sum(w) = 1.
        
        References
        ----------
        Duchi, J., Shalev-Shwartz, S., Singer, Y., & Chandra, T. (2008).
        Efficient projections onto the ℓ1-ball for learning in high dimensions.
        
        Examples
        --------
        >>> v = np.array([0.5, -0.2, 0.8, 0.1], dtype=np.float32)
        >>> w = PortfolioEnv.simplex_projection(v)
        >>> print(w)  # [0.233, 0.0, 0.567, 0.0]
        >>> print(w.sum())  # 1.0
        """
        if v.size == 0:
            return v.astype(np.float32)
        v = np.asarray(v, dtype=np.float32)
        
        # Sort in descending order
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        
        # Find rho: largest index where u[rho] - (sum(u[:rho+1]) - 1)/(rho+1) > 0
        rho = np.nonzero(u * np.arange(1, v.size + 1) > (cssv - 1))[0]
        if rho.size == 0:
            theta = (cssv[-1] - 1.0) / v.size
        else:
            rho = rho[-1]
            theta = (cssv[rho] - 1.0) / float(rho + 1)
        
        # Project and clip
        w = np.maximum(v - theta, 0.0, dtype=np.float32)
        
        # Numerical safety: renormalize if not exactly 1.0
        s = w.sum()
        if s > 0:
            w /= s
        else:
            w[:] = 0.0
        return w

    @staticmethod
    def align_weights(prev_w: npt.NDArray[np.float32],
                      old_assets: List[str],
                      new_assets: List[str]) -> npt.NDArray[np.float32]:
        """
        Map previous weights to new asset universe after monthly rebalancing.
        
        This method handles universe changes that occur at month boundaries or
        when assets enter/exit due to eligibility rules:
        
        - **Survivors**: Assets in both old and new universe keep their weights,
          but normalized to account for exits.
        - **Exits**: Assets in old but not new are liquidated. Their weight is
          redistributed proportionally across survivors (not new entrants).
        - **Entrants**: Assets in new but not old start with 0 weight. The agent
          can allocate to them in the next action.
        
        This logic ensures that forced liquidations (exits) maintain relative
        allocations among survivors, while new entrants don't receive weight
        until the agent explicitly decides to allocate to them.
        
        Parameters
        ----------
        prev_w : np.ndarray
            Shape [A_old], weights at t-1 on old universe.
        old_assets : List[str]
            Asset IDs at t-1 (length A_old).
        new_assets : List[str]
            Asset IDs at t (length A_new).
        
        Returns
        -------
        np.ndarray
            Shape [A_new], dtype float32. Weights aligned to new_assets.
            Satisfies simplex constraint (sum=1, non-negative).
        
        Examples
        --------
        Partial exit (BTC, ETH, XRP) -> (BTC, ETH):
        
        >>> prev_w = np.array([0.5, 0.3, 0.2], dtype=np.float32)
        >>> old = ['BTC', 'ETH', 'XRP']
        >>> new = ['BTC', 'ETH']
        >>> w_new = PortfolioEnv.align_weights(prev_w, old, new)
        >>> print(w_new)  # [0.625, 0.375] (XRP's 0.2 redistributed)
        
        New entrant (BTC, ETH) -> (BTC, ETH, ADA):
        
        >>> prev_w = np.array([0.6, 0.4], dtype=np.float32)
        >>> old = ['BTC', 'ETH']
        >>> new = ['BTC', 'ETH', 'ADA']
        >>> w_new = PortfolioEnv.align_weights(prev_w, old, new)
        >>> print(w_new)  # [0.6, 0.4, 0.0] (ADA starts at 0)
        """
        old_map = {a: i for i, a in enumerate(old_assets)}
        A_new = len(new_assets)
        w_new = np.zeros(A_new, dtype=np.float32)

        # Transfer weights for surviving assets
        survive_indices = []
        survive_weights = []
        for j, a in enumerate(new_assets):
            if a in old_map:
                w_new[j] = prev_w[old_map[a]]
                survive_indices.append(j)
                survive_weights.append(prev_w[old_map[a]])

        surviving_sum = float(np.sum(survive_weights)) if survive_weights else 0.0

        if surviving_sum <= 0.0:
            # All old weights exited: return equal-weight or zeros
            # (Will be reallocated by agent's next action)
            return PortfolioEnv.simplex_projection(w_new)

        # Renormalize survivors to sum=1 (redistributes exited weight)
        # New entrants remain at 0
        w_new = (w_new / surviving_sum).astype(np.float32, copy=False)
        
        # Final projection for numerical safety
        w_new = PortfolioEnv.simplex_projection(w_new)
        return w_new

    def project_to_feasible(self,
                            w_prop: npt.NDArray[np.float32],
                            w_prev: npt.NDArray[np.float32]) -> Tuple[npt.NDArray[np.float32], Dict[str, bool]]:
        """
        Project proposed weights to the feasible set defined by portfolio constraints.
        
        Constraints are enforced in this specific order:
        1. **Non-negativity**: Clip negative weights to 0 (long-only)
        2. **Simplex**: Normalize to sum=1 (fully invested, no cash)
        3. **Per-asset cap**: Clip individual weights (if cfg.max_weight_per_asset set)
        4. **Turnover cap**: Limit L1 distance from previous weights (if cfg.turnover_cap set)
        
        The ordering matters because each projection can violate previous constraints.
        We re-apply simplex projection after caps and use a heuristic for turnover
        (convex blend toward prev_w).
        
        For strict L1 turnover + simplex feasibility, an exact QP solver could be
        used, but the convex blend heuristic is fast and works well in practice.
        
        Parameters
        ----------
        w_prop : np.ndarray
            Shape [A_t], proposed target weights from agent.
        w_prev : np.ndarray
            Shape [A_t], previous realized weights at t-1.
        
        Returns
        -------
        w_exec : np.ndarray
            Shape [A_t], dtype float32. Projected weights satisfying all constraints.
        flags : Dict[str, bool]
            Which constraints were active:
            - 'nonneg': True if negative values were clipped
            - 'simplex': True if normalization was applied (always)
            - 'cap': True if per-asset cap was enforced
            - 'turnover': True if turnover cap was enforced
        
        Examples
        --------
        >>> env = PortfolioEnv(cfg, backend)
        >>> w_prop = np.array([0.8, -0.2, 0.4], dtype=np.float32)  # negative!
        >>> w_prev = np.array([0.33, 0.33, 0.34], dtype=np.float32)
        >>> w_exec, flags = env.project_to_feasible(w_prop, w_prev)
        >>> print(w_exec)  # [0.667, 0.0, 0.333] (negative clipped, renormalized)
        >>> print(flags['nonneg'])  # True
        """
        flags = {"nonneg": False, "simplex": False, "cap": False, "turnover": False}

        if w_prop.size == 0:
            return w_prop.astype(np.float32, copy=False), flags

        w = w_prop.astype(np.float32, copy=False)

        # Step 1: Non-negativity (long-only constraint)
        if (w < 0).any():
            flags["nonneg"] = True
            w = np.maximum(w, 0.0, dtype=np.float32)

        # Step 2: Simplex projection (fully invested, sum=1)
        w = self.simplex_projection(w)
        flags["simplex"] = True  # Always true for numerical safety

        # Step 3: Per-asset concentration cap
        if self.cfg.max_weight_per_asset is not None:
            cap = float(self.cfg.max_weight_per_asset)
            if cap <= 0.0 or cap > 1.0:
                raise ValueError("max_weight_per_asset must be in (0,1].")
            if (w > cap + 1e-12).any():
                flags["cap"] = True
                w = np.minimum(w, cap, dtype=np.float32)
                # Re-project to simplex (capping can reduce sum below 1)
                w = self.simplex_projection(w)

        # Step 4: Turnover cap (L1 distance constraint)
        # This is the most complex constraint because it couples with simplex.
        # We use a simple heuristic: convex blend toward prev_w to satisfy L1 cap.
        # For exact projection, a QP solver would be needed.
        if self.cfg.turnover_cap is not None:
            tau = float(self.cfg.turnover_cap)
            l1_change = float(np.abs(w - w_prev).sum())
            
            if l1_change > tau + 1e-12:
                flags["turnover"] = True
                # Heuristic: blend alpha*w + (1-alpha)*prev_w such that L1 = tau
                # This maintains simplex and reduces L1 distance
                if l1_change > 0:
                    alpha = tau / l1_change
                    w = (alpha * w + (1.0 - alpha) * w_prev).astype(np.float32)
                    # Re-project to simplex for numerical safety
                    w = self.simplex_projection(w)

        return w.astype(np.float32, copy=False), flags

    @staticmethod
    def compute_gross_log_return(w_exec: npt.NDArray[np.float32],
                                 fwd_simple_returns: npt.NDArray[np.float32]) -> float:
        """
        Compute log(w^T * (1 + r_{t+1})).
        Inputs are aligned to the same A_t ordering.
        """
        if w_exec.size == 0:
            return 0.0
        y = 1.0 + fwd_simple_returns.astype(np.float32, copy=False)
        # Safeguard against negatives from bad data (should not happen post-cleaning)
        y = np.maximum(y, 1e-12, dtype=np.float32)
        gross = float(np.dot(w_exec, y))
        return float(np.log(gross))
