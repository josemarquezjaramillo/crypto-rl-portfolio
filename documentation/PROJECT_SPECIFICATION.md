
# Methods: A Unified Crypto Portfolio RL Environment

This document describes the data pipeline, portfolio environment, agent classes, dataset export format, and evaluation protocol used to train and compare reinforcement learning agents for cryptocurrency portfolio management. The three agent families we study are:

- **Policy-gradient with baseline (REINFORCE / A2C / PPO-style)**,
- **Deep Q-Network (DQN-style value-based control)**,
- **Contextual bandit (one-step allocation with arms = portfolios)**.

Our design goals are:
1. No look-ahead leakage (agents never see future data at decision time),
2. Consistent trading constraints across agents,
3. Fair and auditable evaluation under real crypto market regimes,
4. Full reproducibility via frozen dataset exports that can be shared independently of raw market feeds.

These concerns have been repeatedly raised in crypto portfolio RL, where noisy data, regime shifts, and survivorship bias can easily contaminate results [jiang2016drlt, jiang2017eIIE, lucarelli2020dqlcrypto].


---
## 1. Market Data and Investable Universe

### 1.1 Data Source and Horizon
We use daily OHLCV (Open, High, Low, Close, Volume) price/volume bars for liquid cryptocurrencies. Crypto trades 7 days a week, so we represent time as a full calendar of daily timestamps (no weekend gaps).

We ingest raw market data starting **2018-07-01** and continuing through **2025-10-31**. This entire horizon is cleaned and shaped into model-ready tensors.


### 1.2 Warmup Period and Modeling Start
Each agent’s state depends on a rolling **60-day lookback window** of per-asset history. Because of this, we do not allow the agent to make decisions immediately at 2018-07-01. Instead, we treat **2018-07-01 → 2018-08-31** as a warmup/context-only period.

The **first actionable decision day** is **2018-09-01**. All decisions, training steps, and evaluation start no earlier than this date.

This discipline mirrors prior crypto portfolio RL work in which the portfolio policy is conditioned on a trailing tensor of prices/volumes, and the model is not evaluated until such context is available [jiang2017eIIE].


### 1.3 Monthly Membership and Tradable Universe
We define a tradable universe using a **market-cap–weighted index with monthly rebalancing of constituents**:

- At the final trading day of month m−1, we record the index’s membership.
- We freeze that membership for all days in month m.

This prevents any intra-month look-ahead about which assets “should” be in the index and reflects how a systematic allocator would precompute an investable list using only past information [jiang2017eIIE].


### 1.4 Cold-Start Eligibility
An asset i is considered tradable on day t only if:
1. It is included in that month’s index membership, and
2. It has at least **60 consecutive calendar days** of clean, usable OHLCV data immediately prior to t.

This “cold start” requirement prevents the agent from allocating into newly listed or illiquid assets that have not yet established stable trading behavior. Using a fixed-length per-asset lookback window as contextual state is standard in deep RL portfolio managers [jiang2017eIIE, ye2020sarl].


### 1.5 Data Quality: Short-Gap Repair and Reliability Filter
Raw crypto candles can have missing days for individual assets. We correct short gaps in each asset’s OHLCV series after aligning all assets into a wide [date × asset] panel:

- **Gap length = 1 day:** forward fill that single missing bar.
- **Gap length 2–5 days:**
  - Prices (`close`, `high`, `low`) are linearly interpolated between the last observed value before the gap and the first observed value after the gap.
  - Volume is interpolated in `log1p(volume)` space, then exponentiated back and clipped at zero.
- **Gap length > 5 days:** we leave those days as NaN (no fill).

If an asset has such an extended outage (>5 days), it is not eligible to trade until it has again built up at least 60 consecutive clean days. The intuition: tiny outages are data hiccups, 2–5 days can be bridged with reasonable continuity assumptions, but longer gaps often correspond to genuine liquidity or listing issues and should disqualify the asset temporarily [jiang2016drlt, lucarelli2020dqlcrypto].

This rule substantially reduces survivorship bias and overly optimistic continuity assumptions.


---
## 2. State Representation

At each trading decision time t, the environment constructs an observation consisting of (a) a per-asset historical tensor, and (b) the current portfolio context. The agent only sees information available at or before t.


### 2.1 Per-Asset Observation Tensor
For every tradable asset i at time t, we build a lookback window of length 60 calendar days. For each of those 60 days we collect four raw features:

1. Close price
2. High price
3. Low price
4. Volume

We then stack those windows across assets, producing a tensor:

    X_t ∈ ℝ^{A_t × 4 × 60}

where:
- A_t = number of tradable assets at time t, which can vary day by day,
- Channel axis = [Close, High, Low, Volume],
- Temporal axis = most recent 60 daily bars up to and including day t.

The per-asset rolling price history tensor is directly inspired by EIIE-style portfolio RL architectures, which treat each asset as its own “feature map” over recent bars [jiang2017eIIE].


### 2.2 Normalization
We normalize within each asset’s 60-day slice as follows:

- **Price normalization:** Divide `close`, `high`, and `low` at each of the 60 lookback days by the asset’s close on day t. This rescales the series so that the most recent close is 1.0 and all prior prices are relative. This improves stationarity and was shown to stabilize training in crypto portfolio agents [jiang2017eIIE].

- **Volume normalization:**
  - Apply log(1+volume) for each day in the 60-day window.
  - Z-score within that 60-day window (subtract mean, divide by standard deviation).
  - Clip extreme values to [−5, 5].

This volume channel acts as a liquidity/participation signal while preventing large-cap names from dominating purely by magnitude [ye2020sarl].


### 2.3 Variable Universe Size and Asset Ordering
Because membership can change month to month and assets can enter only after a 60-day cold start — and leave if they go illiquid — the number of tradable assets A_t is not fixed.

For each day t we therefore also record an ordered list of tickers (or asset IDs) of length A_t: `asset_list[t]`. The rows of X_t are aligned to this list, so row k in X_t corresponds to `asset_list[t][k]`. This ordering is saved in the exported dataset for reproducibility.

This solves an extremely common source of bugs in financial RL, where changing universes lead to misaligned portfolio weights [jiang2017eIIE, lucarelli2020dqlcrypto].


### 2.4 Portfolio Context and Action Mask
The agent's observation at time t also includes the previous realized portfolio allocation w_{t−1}, i.e. the weights we were actually holding going into day t. This vector is not derivable from OHLCV alone and must be provided by the environment.

Conditioning on w_{t−1} is analogous to the "Portfolio Vector Memory" (PVM) mechanism in [jiang2017eIIE], which helps the policy learn to internalize turnover costs and not churn unnecessarily.

**Optional Action Mask for Batched Training:**  
For batched training with variable A_t across parallel environments, the observation can optionally include an `action_mask` boolean array of fixed size A_max, where True indicates valid assets and False indicates padding positions. This enables efficient vectorized operations in policy-gradient methods without requiring dynamic tensor shapes. The mask is configured via `return_action_mask=True` and `action_mask_size=A_max` in the environment configuration.

Crucially, the agent does not see any information from t+1 or later in its state at time t.


---
## 3. Actions, Execution, and Reward

All agent families act under the same portfolio constraints to ensure comparability.


### 3.1 Portfolio Feasibility and Constraints
On each decision day t, the agent proposes a target allocation w_t across the tradable assets available at time t. We enforce:

- Long-only: w_t(i) ≥ 0 for all assets i.
- Fully invested: sum_i w_t(i) = 1. We explicitly do not include a cash sleeve. This forces the agent to remain allocated to crypto risk rather than trivially “go to cash,” which matters for fair comparison against crypto benchmarks [jiang2017eIIE, lucarelli2020dqlcrypto].
- Per-asset caps: optional caps on single-asset concentration.
- Daily turnover cap: we apply an L1 turnover constraint ||w_t − w_{t−1}||_1 ≤ τ, where τ = 0.30 by default. This models realistic liquidity/impact limits.

All agents are evaluated with the same constraints, so that differences in performance reflect learning behavior and not looser assumptions about trading aggressiveness [lucarelli2020dqlcrypto].


### 3.2 Execution Timing and Transaction Costs
We assume that allocations chosen at the end of day t are executed for the interval [t, t+1]. After rebalancing, we charge proportional transaction costs:

    cost_t = c * ||w_t − w_{t−1}||_1

where c is a slippage/fee parameter. This cost model is common in crypto portfolio RL and in DQN-style crypto trading setups, because it penalizes pathological “rebalance every bar” behavior [jiang2017eIIE, lucarelli2020dqlcrypto].

If the raw proposed allocation w_t_raw violates turnover or concentration constraints, the environment projects it back onto the feasible set and records the projected w_t. The agent is then trained on the executed allocation, not the infeasible proposal.


### 3.3 Reward Definition
After execution, we compute a net reward based on one-day portfolio performance:

    r_net_{t+1} = log(1 + w_t^T R_{t+1}) − c * ||w_t − w_{t−1}||_1

where R_{t+1} is the vector of simple per-asset returns from day t to day t+1 for the assets in `asset_list[t]`. This “growth minus cost” objective follows the EIIE family of portfolio RL methods in crypto [jiang2017eIIE] and is consistent with the RL-for-trading literature [lucarelli2020dqlcrypto].

Importantly, we precompute R_{t+1} for each asset/day pair and store it in the dataset as `fwd_returns[t]`. These forward one-day returns are not part of the observation. The agent never sees them before acting. They are only used by the environment to settle PnL after the action is chosen.


### 3.4 Universe Churn and Forced Liquidations
The investable universe can change:

- Monthly exits: If an asset drops out of the index at a month boundary, we force its weight to 0 at the start of the new month and redistribute that weight proportionally across the remaining assets, charging transaction cost. The agent cannot "pretend" to keep holding delisted assets.

- Monthly new entrants: A new asset entering the index becomes eligible only after it satisfies the 60-day cold-start rule (Section 1.4).

- Intramonth failure / halt: If an asset becomes illiquid or missing for >5 days, we liquidate it at the last reliable close, set its weight to 0, redistribute across remaining assets, and charge cost.

These rules model what a systematic crypto allocator would be forced to do in practice and are consistent with the idea that the environment should remain Markovian in (X_t, w_{t−1}) [jiang2017eIIE, lucarelli2020dqlcrypto].

**Implementation Note:**  
The environment implements this logic via the `align_weights()` static method, which takes the previous portfolio weights and the old/new asset lists, then redistributes exited weights proportionally while initializing new entrants at zero weight. This ensures smooth universe transitions without look-ahead bias.


---
## 4. Agent Classes (Planned Architectures)

**Note:** This section describes the planned agent architectures for the project. The environment implementation (`environment/environment.py`) provides the common MDP interface that all agents will use. Agent implementations are scheduled for future development.

All agents interact with the same environment API. The only difference is how they choose an allocation w_t.


### 4.1 Policy-Gradient with Baseline (REINFORCE / A2C / PPO-style)
A policy network consumes the observation at time t — including the per-asset tensor X_t and the previous allocation vector w_{t−1} — and outputs unnormalized logits over the current tradable assets.

We apply a mask so assets that are not tradable on day t (because of membership/cold-start rules) receive −∞ logit. We then apply a softmax over the masked logits to produce a continuous allocation w_t on the simplex. After that, we apply the turnover/concentration projection described above.

This approach directly parameterizes the portfolio weights and is similar in spirit to the EIIE / PVM structure for crypto portfolio management [jiang2017eIIE] and to policy-gradient market-making work in crypto order books [sadighian2019mmppo]. The baseline / critic (A2C, PPO) provides variance reduction for the gradient estimate.


### 4.2 Deep Q-Network (DQN)
A DQN requires a discrete action space. We therefore define a catalog of feasible portfolios (candidate allocations) for the current universe. Examples in the catalog include:
- equal-weight among the top K assets by cap,
- sparse 2- or 3-asset mixes with capped weights,
- diversified allocations that obey turnover and per-asset caps.

At each step, the DQN chooses one catalog element. The environment then enforces constraints and executes that allocation.

This gives the DQN a tractable yet realistic action space. It mirrors how deep Q-learning has been applied to crypto trading strategies and allocation heuristics, where Q-values correspond to discrete trading/positioning choices [mnih2015dqn, lucarelli2020dqlcrypto].


### 4.3 Contextual Bandit
We treat each catalog portfolio (the same catalog used by DQN) as an “arm.” The bandit observes the current state (or an embedding of it), and selects which arm to deploy for day t.

This can be implemented as:
- discounted Thompson Sampling / UCB on arm statistics, or
- a neural contextual bandit that outputs arm scores given the state.

This formulation aligns with “bandit networks” for portfolio selection under nonstationary returns and risk, where each allocation is a competing arm and the goal is to adaptively switch among them [huo2017riskbandit, fonseca2024banditnets].

The contextual bandit does not explicitly optimize long-horizon value functions. Instead, it treats each day’s allocation choice as an immediate reward maximization problem, which can be a strong baseline in highly nonstationary markets.


---
## 5. Dataset Construction and Export

A major part of this work is to produce a frozen, versioned dataset that can be shared and reproduced. We do not assume access to proprietary feeds at training time. Instead, we pre-build `dataset_v1/` and train all agents against it.


### 5.1 Time Splits
We split calendar time into three segments:

1. Warmup / context only:  
   2018-07-01 → 2018-08-31  
   Used solely to build the first valid 60-day lookback windows. No actions are taken, and these dates do not appear in train/val/test metrics.

2. Development (Dev) period:  
   2018-09-01 → 2023-12-31  
   This includes training data and validation data. All hyperparameter tuning and model selection happens here.

3. Final Test (Out-of-Sample) period:  
   2024-01-01 → 2025-10-31  
   Models are frozen before entering this period. No parameter updates or hyperparameter changes are allowed here. This is the headline out-of-sample evaluation [jiang2016drlt, lucarelli2020dqlcrypto].


### 5.2 Regime-Based Validation Windows
Within the Dev period, we do not rely on a single contiguous validation split (which can bias the agent toward whatever regime that slice happened to be in). Instead, we carve out several ~20-day validation windows that correspond to qualitatively distinct crypto regimes, such as:
- crash / forced deleveraging,
- liquidity shock,
- runaway bull,
- grinding bear,
- low-vol chop.

Each of these ~20-day windows is assigned a split_tag like `val_window_2020_covid`. All other Dev dates are tagged `train_core`. The agent’s hyperparameters are selected based on average performance across all validation windows, not just one regime [jiang2016drlt, jiang2017eIIE, lucarelli2020dqlcrypto].

This prevents cherry-picking “the good period” for tuning and explicitly acknowledges regime nonstationarity in crypto.


### 5.3 Per-Day Records
For each actionable decision day t in Dev or Test, we record:

- `obs_tensor[t]`: float32 array of shape [A_t, 4, 60], containing the normalized OHLCV lookback window for each currently tradable asset (Section 2).
- `asset_list[t]`: ordered list of asset tickers / IDs of length A_t. This fixes the row-to-asset mapping for that day.
- `fwd_returns[t]`: float32 array of shape [A_t], where entry i is the simple return from day t to t+1 for asset i. These forward returns are used by the environment to compute realized portfolio PnL and reward, but they are never exposed to the agent at decision time.
- `split_tag[t]`: one of "train_core", "val_window_k", or "test".

Because A_t can change over time, these structures are stored per day and aligned by `asset_list[t]`.


### 5.4 Export Format (`dataset_v1/`)
We export the full dataset as a versioned directory, e.g. `dataset_v1/`, containing:

- `metadata.json`  
  Global constants and experiment settings, including:
  - Lookback length (60 days),
  - Turnover cap (τ = 0.30),
  - Gap-repair policy (forward fill 1 day, interpolate ≤5 days, else NaN),
  - Long-only / fully invested / no cash sleeve,
  - Dev/Test calendar boundaries,
  - The list of validation windows and their date ranges.

- `dev_index.parquet` and `test_index.parquet`  
  Tidy tables with columns:
  - `date` (Timestamp),
  - `split_tag` (e.g. train_core, val_window_2020_covid, or test).
  These define the ordered decision timeline for each split.

- `dev_obs_tensors.npz` and `test_obs_tensors.npz`  
  Compressed NumPy archives. Keys are strings like "t_2021-06-15". Values are the per-day [A_t, 4, 60] observation tensors.

- `dev_asset_lists.jsonl` and `test_asset_lists.jsonl`  
  One line per day, e.g.:
  {"date": "2021-06-15", "assets": ["BTC","ETH", "..."]}
  This preserves the asset ordering for each day.

- `dev_fwd_returns.npz` and `test_fwd_returns.npz`  
  Compressed NumPy archives containing per-day forward simple returns vectors aligned to `asset_list[t]`.

This export is the canonical research artifact. You can zip `dataset_v1/` and share it with collaborators (e.g. via cloud storage). They do not need to run the entire raw data pipeline to reproduce experiments.


---
## 6. Code Architecture and Reproducibility Boundary

To make experiments auditable and sharable, we explicitly separate the pipeline into four modules. This also defines the “reproducibility boundary.”

### 6.1 `data_loader.py`
- Loads raw OHLCV and index membership data into pandas DataFrames.
- Normalizes calendars (daily index, 7 days/week).
- Does not enforce RL rules, portfolio assumptions, or eligibility logic.
- Think of this as “market ingest.”

### 6.2 `data_builder.py`
- Applies all research assumptions and transforms raw data into model-ready tensors:
  - Gap repair and interpolation rules (forward fill 1 day, interpolate up to 5 days, otherwise NaN),
  - Rolling 60-day lookback window,
  - Monthly index membership frozen within each month,
  - 60-day cold-start eligibility for new assets,
  - Forced removal of assets with long data gaps,
  - Computation of forward one-day returns for reward.
- Produces three aligned structures, indexed by date t:
  - `obs_tensors[t]`: [A_t, 4, 60] normalized OHLCV window,
  - `asset_lists[t]`: list of tradable assets at t,
  - `fwd_returns[t]`: next-day returns aligned with `asset_lists[t]`.

This step encodes the assumptions standard in crypto portfolio RL: conditioning on trailing normalized price/volume tensors [jiang2017eIIE, ye2020sarl], enforcing realistic tradability, and computing ex-post rewards without leakage [lucarelli2020dqlcrypto].


### 6.3 `data_exporter.py`
- Takes the per-day outputs from `data_builder.py` and organizes them into the official experiment splits:
  - Warmup (2018-07-01 → 2018-08-31, context only),
  - Dev (2018-09-01 → 2023-12-31),
  - Test (2024-01-01 → 2025-10-31).
- Tags each Dev day as either train_core or one of several ~20-day regime-specific validation windows (e.g. crash, liquidity shock, 2021 bull, 2022 deleverage, 2023 chop). This supports robust hyperparameter selection across regimes rather than overfitting to a single contiguous block [jiang2016drlt, jiang2017eIIE, lucarelli2020dqlcrypto].
- Writes out `dataset_v1/` exactly as described in Section 5.4, including metadata.json.

This is the reproducibility boundary. After `data_exporter.py` runs, we have a frozen dataset (`dataset_v1/`) that fully specifies the training, validation, and test timelines and observations.


### 6.4 `dataset_loader.py`
- Loads a dataset snapshot like `dataset_v1/` back into memory.
- Returns an `ExportedDataset` object with:
  - `.index_df` — rows of (date, split_tag) in order,
  - `.obs_tensors[date]` — the [A_t, 4, 60] tensor for that date,
  - `.asset_lists[date]` — the tradable assets in row order for that date,
  - `.fwd_returns[date]` — the forward one-day return vector for that date,
  - `.metadata` — global experiment settings (lookback length, turnover cap, etc.).
- Provides helper methods like `.dates()` and `.get_day("YYYY-MM-DD")` for convenient environment stepping.

Critically: all RL training/evaluation code (environments, agents, baselines) uses only `dataset_loader.py` and a frozen dataset directory (e.g. `dataset_v1/`). No one touches raw OHLCV or re-runs preprocessing during experiments.

This guarantees that results are reproducible and auditable, and it allows collaborators to run the exact same experiments simply by downloading `dataset_v1/` and pointing the environment code at it [jiang2017eIIE, lucarelli2020dqlcrypto].


### 6.5 `dataset_backend.py`
- Adapter that bridges `ExportedDataset` (from `dataset_loader.py`) to `PortfolioEnv` (the RL environment).
- Handles type conversions: string dates (from Parquet index) ↔ `np.datetime64[D]` (expected by environment).
- Unpacks dictionary format (returned by `ExportedDataset.get_day()`) to tuple format `(features, asset_ids, fwd_returns)` expected by the environment.
- Implements split tag filtering: accepts `split_tag_filter` parameter (e.g., `"train_core"`, `["val_window_val_bear", "val_window_val_chop"]`) to subset the dataset for specific training/validation scenarios.
- Returns a `DataBackend` interface implementation that the environment can use without knowing the underlying data storage format.

This adapter pattern follows SOLID principles by separating data loading concerns from environment logic, making the codebase more maintainable and testable. The environment never directly touches `ExportedDataset` or Parquet files—it only interacts with the abstract `DataBackend` interface.


---
## 7. Evaluation Protocol

### 7.1 Training, Hyperparameter Tuning, and Freezing
1. We train candidate agents on Dev days tagged train_core.
2. We evaluate those trained candidates — with parameters frozen — on each Dev validation window (val_window_*). These ~20-day validation windows are spread across very different market regimes (crash, bull, deleverage, chop), which prevents tuning only for a single favorable regime [jiang2016drlt, lucarelli2020dqlcrypto].
3. We select hyperparameters that do well across these validation regimes, not just one.
4. We then retrain the final model from scratch on the entire Dev period (train_core + all validation windows together) using the chosen hyperparameters. This yields a final policy snapshot for that agent class.

This mirrors how crypto RL studies report strategies over multiple disjoint subperiods to illustrate robustness [jiang2016drlt, jiang2017eIIE, lucarelli2020dqlcrypto].


### 7.2 Final Out-of-Sample Test
We evaluate the frozen final policy on the Test period 2024-01-01 → 2025-10-31, which is strictly out-of-sample. During Test:
- The agent cannot update its parameters,
- The hyperparameters are locked,
- The feasible universe and constraints still evolve realistically (assets can leave/enter, turnover and cost still apply),
- We measure performance over time exactly as the environment would experience it in deployment.

This walk-forward evaluation structure is standard in financial RL and algorithmic trading: you treat a later period of data that the agent never saw as the true exam [lucarelli2020dqlcrypto].


### 7.3 Metrics
We report:
- Annualized Return,
- Annualized Volatility,
- Sharpe Ratio,
- Sortino Ratio,
- Max Drawdown,
- Calmar Ratio,
- Turnover (average ||w_t − w_{t−1}||_1),
- Hit Rate (fraction of profitable days),
- Optional per-regime breakdowns within the Test period (e.g. first half of 2024 vs second half of 2024).

These are standard in RL-for-trading work and in crypto trading evaluations, where both raw return and risk/instability (drawdown, turnover) matter [jiang2016drlt, lucarelli2020dqlcrypto].


### 7.4 Experiment Tracking and Logging
The environment supports optional CSV logging via the `log_dir` parameter in `EnvConfig`. When enabled, each episode writes a timestamped CSV file with columns:

- **Temporal**: `step`, `date`
- **Universe**: `n_assets` (number of tradable assets on that day)
- **Trading metrics**: `turnover` (L1 norm of weight changes), `transaction_cost`
- **Performance**: `gross_log_return`, `reward_net`, `portfolio_value`
- **Constraint diagnostics**: `constraint_nonneg`, `constraint_simplex`, `constraint_cap`, `constraint_turnover` (boolean flags indicating which constraints were active during projection)

Log files are named with the pattern `env_{split}_{seed}_{timestamp}.csv` and are automatically flushed to disk every 10 steps and on environment close. This facilitates post-hoc analysis, hyperparameter debugging, and ablation studies without requiring custom logging code in each agent implementation.


---
## 8. Why This Setup is Defensible

1. No look-ahead leakage.  
   The agent’s observation at t includes only data available by the end of day t. The forward return vector for day t→t+1 is stored separately (`fwd_returns[t]`) and is only used after the action to compute reward. This matches best practice in portfolio RL [jiang2017eIIE, lucarelli2020dqlcrypto].

2. Realistic constraints.  
   All agents must produce long-only, fully invested portfolios with no explicit cash sleeve, are penalized for turnover, and pay transaction costs. This prevents “cheating” via sitting in cash or overtrading and makes evaluation comparable to real crypto allocation and to baseline strategies [jiang2017eIIE, lucarelli2020dqlcrypto].

3. Universe governance.  
   Monthly membership using a market-cap–weighted index, plus a 60-day cold-start requirement and de-listing logic, means the agent only trades assets that a real allocator could reasonably include. This mitigates survivorship bias and unrealistic exposure to newly listed or dead assets [jiang2017eIIE, ye2020sarl].

4. Regime-aware validation.  
   Instead of tuning on one contiguous validation slice, we define multiple ~20-day validation windows across drastically different crypto regimes (crash, spike, deleverage, chop). Hyperparameters are chosen to perform across all of them, reflecting the known regime instability of crypto returns [jiang2016drlt, lucarelli2020dqlcrypto].

5. Walk-forward out-of-sample test.  
   The final evaluation period (2024-01-01 → 2025-10-31) is strictly held out. Agents are frozen going in, and performance there represents the actual deploy-time scenario in a live system [lucarelli2020dqlcrypto].

6. Explicit reproducibility boundary.  
   We generate a frozen dataset snapshot (`dataset_v1/`) using `data_exporter.py` and only then train and evaluate agents by loading that snapshot via `dataset_loader.py`. This makes experiments auditable and shareable: another researcher can reproduce our results just by loading `dataset_v1/`, without touching any proprietary data feeds or re-running preprocessing [jiang2017eIIE, lucarelli2020dqlcrypto].


---
## 9. Code Quality and Testing

The environment implementation (`environment/environment.py`) is validated by a comprehensive pytest test suite (`tests/test_environment.py`) covering:

**Unit tests:**
- Simplex projection (Duchi et al. algorithm): Tests for already-valid inputs, negative values, uniform values, edge cases (empty arrays, single element)
- Weight alignment across universe changes: Tests for no-change scenarios, partial exits, new entrants, complete turnover, asset reordering
- Constraint projection: Tests for non-negativity clipping, per-asset caps, turnover limits, already-feasible cases

**Integration tests:**
- Environment initialization and reset with real dataset artifacts
- Single-step execution with proper observation/reward/info structure
- Full episode execution (50+ steps) with constraint enforcement
- Deterministic seeding and reproducibility
- Terminal condition handling

**Feature tests:**
- CSV logging: File creation, schema validation, correct flushing
- Action mask: Shape, dtype, correct True/False patterns for valid/padding assets
- Backward compatibility: All features optional and non-breaking

**Edge case tests:**
- Empty date ranges (should raise ValueError)
- Step after episode termination (should raise RuntimeError)
- Invalid configuration parameters (proper validation)

**Performance tests:**
- Step speed benchmarking (< 5 seconds for 100 steps on modern hardware)

Run the test suite with:
```bash
pytest tests/test_environment.py -v
```

All 31 tests pass, ensuring the environment is production-ready for agent development.

**Smoke test demonstration:**  
A comprehensive end-to-end demonstration script (`smoke_test.py`) shows:
- Complete workflow: Load dataset → Create backend → Initialize environment → Train agents → Evaluate
- Baseline agents (Random, Uniform/1-N) for sanity checking
- Multiple usage examples: continuous actions (A2C/PPO), discrete actions (DQN), CSV logging, action masks
- Training and validation evaluation with realistic metrics

Run the smoke test with:
```bash
python smoke_test.py
```

This script serves as both a validation tool and living documentation for new users.


---
## References

- [jiang2016drlt] Jiang, Z. (2016). "Cryptocurrency Portfolio Management with Deep Reinforcement Learning." arXiv:1612.01277.
- [jiang2017eIIE] Jiang, Z., Xu, D., & Liang, J. (2017). "A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem (EIIE / PVM)." arXiv:1706.10059.
- [ye2020sarl] Ye, Y., Zhang, X., Zhang, L., Wang, H., & Wang, D. (2020). "State Augmented Reinforcement Learning for Portfolio Management." AAAI / arXiv:2002.05780.
- [lucarelli2020dqlcrypto] Lucarelli, G., & Borrotti, M. (2020). "Deep Reinforcement Learning for Cryptocurrency Trading." Neural Computing and Applications.
- [mnih2015dqn] Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). "Human-level control through deep reinforcement learning." Nature.
- [huo2017riskbandit] Huo, H., & Fu, M. C. (2017). "Risk-aware multi-armed bandit and portfolio selection." Royal Society Open Science, 4(1), 160641.
- [fonseca2024banditnets] de Freitas Fonseca, P., et al. (2024). "Improving Portfolio Optimization Results with Bandit Networks." arXiv:2410.04217.
- [sadighian2019mmppo] Makridakis, J., et al. (2019). "Deep Reinforcement Learning for Cryptocurrency Market Making (A2C/PPO)." arXiv preprint.
