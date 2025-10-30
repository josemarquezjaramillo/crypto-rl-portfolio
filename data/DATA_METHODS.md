# Methods: A Unified Crypto Portfolio RL Environment

This section documents the data pipeline, environment design, state/action definitions, reward structure, dataset splits, and evaluation setup for a family of cryptocurrency portfolio managers. We evaluate three agent classes under a shared environment:
- **Value-based control (DQN-style)**,
- **Policy-gradient with baseline (REINFORCE / A2C / PPO-style)**,
- **Contextual bandit (one-step allocation)**.

We emphasize: (1) no look-ahead leakage, (2) consistent constraints across agents, and (3) robustness to crypto market regime shifts. These concerns are repeatedly highlighted in prior work on deep RL portfolio trading and crypto allocation. citejiang2016drltjiang2017eIIElucarelli2020dqlcrypto

---

## 1. Market Data and Investable Universe

### 1.1 Data Source and Horizon
We use daily OHLCV (Open, High, Low, Close, Volume) data for liquid cryptocurrencies. Crypto trades on a continuous 7-day calendar, so we model time as a contiguous daily index without weekend gaps. Our raw data horizon spans **2018-07-01 through 2025-10-31**.

### 1.2 Warmup Period and Modeling Start
Agents require a fixed-length historical window to form state. We use a **60-day lookback** window. Because of that, although we load data beginning **2018-07-01**, we only allow the first actionable decision on **2018-09-01**. Dates before 2018-09-01 are used solely to construct valid context windows and are never used to score performance or to let the agent act. This “warmup then start” pattern matches portfolio RL work where the agent’s observation is a trailing tensor of market history (e.g. Jiang et al. 2017). citejiang2017eIIE

### 1.3 Monthly Membership and Tradable Universe
We define an investable universe via a **market-cap–weighted index with monthly rebalancing**. On the final trading day of month \(m-1\), we record the index constituents, and we hold that membership fixed for all days in month \(m\). This prevents look-ahead (no using future composition intra-month) and is analogous to the way Jiang et al. select assets based only on historical liquidity information prior to a backtest window. citejiang2017eIIE

### 1.4 Cold-Start Eligibility Rule
An asset can only be traded on day \(t\) if:
1. It is in the index membership for that month, and
2. It has at least **60 consecutive calendar days** of usable OHLCV history immediately preceding \(t\).

This “cold-start” rule prevents the agent from allocating into assets with incomplete, noisy, or obviously biased early history. Using a fixed lookback window as the model’s per-asset context is standard in deep portfolio RL. citejiang2017eIIEye2020sarl

### 1.5 Short-Gap Data Repair and Reliability Filter
Crypto data feeds occasionally miss daily bars for an asset. We correct gaps in each asset’s OHLCV time series **after pivoting to a wide [date × asset] panel**:

- **Single missing day (gap length = 1):** we forward fill that missing bar.
- **Short outage (2–5 consecutive missing days):**  
  - For prices (`close`, `high`, `low`): we linearly interpolate between the last observed value before the gap and the first observed value after the gap.  
  - For volume: we interpolate in log space (`log1p(volume)`), then back-transform, and clip negatives to zero.
- **Extended outage (>5 days):** we do **not** fill. Those days stay NaN.

If an asset has a long outage, it fails eligibility until it re-establishes a full 60-day clean history. Intuitively: tiny gaps are data hiccups, 2–5 day gaps are plausible continuity, but anything longer implies delisting / severe illiquidity and should not be treated as continuous tradability.

This is stricter than naive forward-fill, which can hallucinate stability, and makes our dataset robust against survivorship and exchange feed noise — a recurring concern in crypto RL backtests. citejiang2016drltlucarelli2020dqlcrypto

---

## 2. State Representation

At each decision day \(t\), the environment produces a **per-asset observation tensor** and environment context. The agent only sees information available \(\le t\).

### 2.1 Per-Asset Observation Tensor
For each tradable asset \(i\) at time \(t\), we build a 60-day window of features:
- **Close price**
- **High price**
- **Low price**
- **Volume**

We stack those into a tensor of shape \([A_t, 4, 60]\), where:
- \(A_t\) is the number of tradable assets on day \(t\) (after monthly membership + cold-start screening),
- feature channels are \([C, H, L, V]\),
- the temporal axis is the trailing 60 calendar days up to and including \(t\).

This mirrors the “price tensor per asset over a recent lookback horizon” design used by Jiang et al. (EIIE), which feeds a stack of recent normalized bars to a CNN-like portfolio policy. citejiang2017eIIE

### 2.2 Normalization
- **Price normalization:** For each asset, we divide `close`, `high`, and `low` by the asset’s close at day \(t\). That makes the most recent close equal to 1.0 and expresses the recent path in relative terms, which stabilizes training and was shown to work well in EIIE-style crypto portfolio agents. citejiang2017eIIE  
- **Volume normalization:** For each asset in the 60-day window, we apply \(\log(1+\text{volume})\), then z-score within that window (subtract mean, divide by std), and clip to \([-5, 5]\). This gives a liquidity/participation signal while preventing huge-cap assets from trivially dominating purely on scale. Volume-like context has been used as an additional conditioning channel in crypto RL variants and state-augmentation work. citeye2020sarl

### 2.3 Portfolio Context
The observation at time \(t\) is augmented with the **previous portfolio allocation** \(w_{t-1}\), which is the vector of realized portfolio weights held going into decision \(t\). This is not inferable from raw OHLCV and must be supplied by the environment.

This mirrors the “Portfolio Vector Memory” (PVM) idea in Jiang et al. (2017), where the agent conditions on its previous allocation to learn to internalize transaction costs and avoid pathological turnover. citejiang2017eIIE

We do **not** bake forward returns or any post-\(t\) data into the state. The agent never sees tomorrow’s price information at decision time.

---

## 3. Action Space, Execution, and Reward

### 3.1 Portfolio Feasibility
We enforce realistic allocation constraints consistent across all agent families:
- Long-only,
- Fully invested,
- No cash sleeve, i.e. \(\sum_i w_{t}(i) = 1\), \(w_{t}(i) \ge 0\),
- Per-asset caps (e.g. prevent 100% into one illiquid coin),
- Daily **turnover cap** of the form \(\|w_t - w_{t-1}\|_1 \le \tau\), with \(\tau = 0.30\) by default.

We intentionally exclude an explicit “cash asset.” Cash-heavy solutions can trivially deflate risk and dominate Sharpe on paper, but that’s not comparable to our benchmarks (which are fully invested crypto indices). We want to evaluate real allocation skill, not “go to cash.” This matches the comparison philosophy in crypto RL and bandit-style crypto allocation work, where the agent’s job is to choose among risky assets, not market-time to fiat. citejiang2017eIIEhuo2017riskbanditlucarelli2020dqlcrypto

### 3.2 Execution Timing and Transaction Costs
At the end of day \(t\), the agent proposes a portfolio \(w_t\). We assume that order is executed for day \(t+1\) and that realized PnL over \([t, t+1]\) is determined by that allocation.

We apply proportional transaction costs of the form
\[
\text{cost}_t = c \cdot \lVert w_t - w_{t-1} \rVert_1,
\]
where \(c\) is a slippage/fee parameter. EIIE explicitly optimizes portfolio growth net of transaction costs, and DQN-style crypto trading work also accounts for frictions to avoid degenerate “trade every bar” behavior. citejiang2017eIIElucarelli2020dqlcrypto

We also enforce the turnover cap \(\|w_t - w_{t-1}\|_1 \le \tau\). In practice, the environment projects any proposed allocation back into the feasible set if needed. This projection is recorded so the agent learns from the *actual* executed allocation, not the raw proposal.

### 3.3 Reward
After execution, we compute the **net log return**:
\[
r^{\text{net}}_{t+1}
=
\log\!\big(1 + w_t^\top R_{t+1}\big)
-
c \, \lVert w_t - w_{t-1} \rVert_1,
\]
where \(R_{t+1}\) is the vector of simple per-asset returns from day \(t\) to \(t+1\).

This form (growth minus cost) is essentially what Jiang et al. optimize with EIIE (portfolio value increment with cost), and is common in actor-critic and DQN-style financial RL. citejiang2017eIIElucarelli2020dqlcrypto

We precompute \(R_{t+1}\) for every asset/day pair and store it in the dataset as `fwd_returns[t]`. Crucially, `fwd_returns[t]` is *never included in the observation at t*. It’s only used after the action to settle PnL and produce the reward.

### 3.4 Universe Churn and Forced Liquidations
- **Monthly exits:** If an asset leaves the index at a monthly boundary, its weight is forcibly set to 0 at the start of the new month and reallocated proportionally across remaining assets. We charge transaction cost for that unwind.  
- **Monthly new entrants:** A new index member only becomes tradable after it has satisfied the 60-day cold-start requirement.  
- **Intramonth delist / halt:** If an asset stops having reliable data (e.g. vanishes for >5 days), it becomes instantly ineligible: we liquidate it at the last reliable close and redistribute. The turnover cost applies.

This mimics how a realistic systematic crypto allocator must deal with listings, delistings, and liquidity collapses, and it keeps the process Markov in \((X_t, w_{t-1})\). citejiang2017eIIElucarelli2020dqlcrypto

---

## 4. Agents

All agent classes consume the same environment interface and constraints, but act differently.

### 4.1 Policy-Gradient with Baseline (REINFORCE / A2C / PPO-style)
The policy network outputs logits over the tradable assets at time \(t\). We apply a mask to disallow ineligible assets (those not in the month’s membership or failing cold-start). Softmax over the masked logits yields a continuous allocation \(w_t\) on the simplex. We then enforce turnover and weight caps via projection.

This matches “learned portfolio weight maps directly from state,” as in actor-critic crypto portfolio methods and the PVM+softmax allocation in EIIE. citejiang2017eIIEmakridakis2019mmppo  
The baseline/critic (A2C/PPO style) reduces variance in policy gradient estimates.

### 4.2 Deep Q-Network (DQN)
DQN requires a discrete action space. We define a **catalog of feasible portfolios** (e.g. sparse allocations, top-\(K\) equal-weight bundles, capped weight vectors) that respect long-only, fully invested, turnover cap from \(w_{t-1}\), and current membership. At each step, the DQN selects one catalog element. The environment executes that portfolio (after projection if necessary).

Deep Q-learning has been applied directly to crypto trading using replay buffers, target networks, and epsilon-greedy exploration. citemnih2015dqnlucarelli2020dqlcrypto  
Our catalog approach gives DQN a realistic yet tractable action space and ensures fairness: all candidate portfolios obey the same constraints the policy-gradient agent faces.

### 4.3 Contextual Bandit
We treat each catalog portfolio (same catalog as DQN) as an “arm.” The bandit selects one arm each day based on the current state embedding (context). We consider discounted Thompson Sampling / UCB-style approaches and “bandit networks” for nonstationary portfolio allocation. citehuo2017riskbanditfonseca2024banditnets

This produces a one-step allocation policy without explicit long-horizon credit assignment, which is a common framing when you treat daily portfolio choice as a contextual bandit under regime drift.

---

## 5. Dataset Construction and Export

A core contribution of this work is that **we freeze a reproducible dataset** for training, validation, and final evaluation. This prevents silent data leakage and makes experiments auditable and shareable.

### 5.1 Timeline Splits
We define three relevant horizons:

1. **Warmup (context only; no actions):**  
   2018-07-01 → 2018-08-31  
   Used solely to fill the first 60-day windows.

2. **Development (Dev) period:**  
   **2018-09-01 → 2023-12-31**  
   This is the pool for training and hyperparameter tuning.

3. **Final Test period (Out-of-Sample):**  
   **2024-01-01 → 2025-10-31**  
   Models are frozen before this period begins. No learning or tuning happens here.

This mirrors how financial RL work typically reserves a structurally later period as a true out-of-sample walk-forward evaluation to avoid regime overfitting. citejiang2016drltlucarelli2020dqlcrypto

### 5.2 Regime-Based Validation Windows
Inside the Dev period, we designate several ~20-day **validation windows** that correspond to distinct market regimes (e.g. crash, panic, melt-up, deleverage, chop). These discontiguous validation windows are used only for hyperparameter/model selection; the agent does **not** update its weights while being evaluated on them.

All other Dev days are `"train_core"`. Each day in Dev is thus tagged as either `"train_core"` or `"val_window_*"`. This approach addresses the criticism that tuning on one contiguous block (e.g. just “the 2021 bull run”) leads to regime bias. citejiang2016drltjiang2017eIIElucarelli2020dqlcrypto

### 5.3 Test Tagging
Every decision day in the final test horizon (2024-01-01 → 2025-10-31) is tagged `"test"`. These timestamps are strictly off-limits for training or hyperparameter selection.

### 5.4 Per-Day Records
For every decision day \(t\) in Dev or Test we store:

- `obs_tensor[t]`: float32 array of shape \([A_t, 4, 60]\), the normalized OHLCV history for each eligible asset.  
- `asset_list[t]`: ordered list of asset identifiers, length \(A_t\), aligned to rows in `obs_tensor[t]`.  
- `fwd_returns[t]`: float32 array of shape \([A_t]\), containing next-day simple returns for those assets, used by the environment to compute realized portfolio PnL and reward.  
- `split_tag[t]`: `"train_core"`, `"val_window_k"`, or `"test"`.

In addition, we store global metadata:
- The lookback length (60 days),
- The turnover cap (\(\tau = 0.30\) in L1),
- The gap-repair policy (≤1-day ffill, ≤5-day interpolate, otherwise NaN),
- The fact that we enforce long-only, fully invested, no cash sleeve,
- The exact Dev/Test date boundaries,
- The specific validation window date ranges and names.

### 5.5 Export Format
We checkpoint the dataset to a versioned directory, e.g. `dataset_v1/`, containing:

- `metadata.json` — experiment constants, validation window definitions, constraints.  
- `dev_index.parquet` / `test_index.parquet` — for each date, the split tag.  
- `dev_obs_tensors.npz` / `test_obs_tensors.npz` — compressed numpy arrays keyed by date (e.g. `"t_2021-06-15"`).  
- `dev_asset_lists.jsonl` / `test_asset_lists.jsonl` — asset universe per day (human-readable).  
- `dev_fwd_returns.npz` / `test_fwd_returns.npz` — forward one-day returns per asset per day.

This snapshot is directly shareable (e.g. zipped and dropped in cloud storage), and lets collaborators or reviewers recreate experiments exactly, without re-running any preprocessing or eligibility logic. That in turn prevents accidental leakage, inconsistent data cleaning, or subtle differences in index membership rules across machines — a known reproducibility pitfall in financial RL. citejiang2017eIIElucarelli2020dqlcrypto

---

## 6. Evaluation Protocol

### 6.1 Training / Tuning / Freezing
- We train agents on Dev `"train_core"` days.
- We score candidate hyperparameters on each `"val_window_*"` block (agents are frozen during scoring). We select hyperparameters that perform well across these heterogeneous regimes, not just in one market condition.
- After hyperparameters are fixed, we retrain the agent once on the entire Dev period (train_core + validation windows together) to produce a final policy snapshot.
- That snapshot is **frozen**.

This mirrors what people actually do in practice for financial RL: use historical data to shape the model, but then produce a single “deployed” policy to evaluate on true out-of-sample. citelucarelli2020dqlcrypto

### 6.2 Final Out-of-Sample Test
We evaluate the frozen policy on the `"test"` period (2024-01-01 → 2025-10-31). No learning, no hyperparameter changes, no catalog redesign for DQN except what is mechanically required by index membership rules and turnover constraints. This is the headline performance.

### 6.3 Metrics
We report:
- Annualized Return,
- Annualized Volatility,
- Sharpe Ratio,
- Sortino Ratio,
- Max Drawdown,
- Calmar Ratio,
- Turnover,
- Hit Rate (fraction of profitable days),
- Per-regime breakdowns in the test window.

These are standard in RL-for-trading work and DQN crypto trading studies, where Sharpe, drawdown, and turnover are central to evaluating whether “intelligent allocation” is happening or the model is just levering momentum at the cost of ruin risk. citejiang2016drltlucarelli2020dqlcrypto

---

## 7. Summary of Why This Setup Is Defensible

1. **No leakage:** The agent only sees data \(\le t\). Forward returns are stored separately and only used after the action to settle reward, following best practice in portfolio RL. citejiang2017eIIE  
2. **Realistic constraints:** Long-only, fully invested, no “cheat cash,” turnover caps, transaction costs, and forced liquidations on delist/exit match crypto allocator reality. citejiang2017eIIElucarelli2020dqlcrypto  
3. **Universe governance:** Monthly membership from a cap-weighted index, plus a 60-day cold-start requirement, prevents survivorship bias and avoids giving the agent access to assets that didn’t have enough trading history. citejiang2017eIIE  
4. **Stress testing across regimes:** We explicitly carve out multiple discontiguous validation windows across very different market regimes, inspired by prior RL crypto portfolio studies that emphasize regime instability. citejiang2016drltjiang2017eIIElucarelli2020dqlcrypto  
5. **Replicability:** We export a fully specified, versioned dataset (`dataset_v1/`) including tensors, asset lists, forward returns, split tags, and metadata.json. This allows collaborators (and reviewers) to exactly reproduce train/val/test behavior and results.

---

## References (Inline Keys)

- [jiang2016drlt] Jiang, Z. (2016). *Cryptocurrency Portfolio Management with Deep Reinforcement Learning.* arXiv:1612.01277.  
- [jiang2017eIIE] Jiang, Z., Xu, D., & Liang, J. (2017). *A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem (EIIE / PVM).* arXiv:1706.10059.  
- [ye2020sarl] Ye, Y., Zhang, X., Zhang, L., Wang, H., & Wang, D. (2020). *State Augmented Reinforcement Learning for Portfolio Management.* AAAI / arXiv:2002.05780.  
- [lucarelli2020dqlcrypto] Lucarelli, G., & Borrotti, M. (2020). *Deep Reinforcement Learning for Cryptocurrency Trading.* Neural Computing and Applications.  
- [mnih2015dqn] Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). *Human-level control through deep reinforcement learning.* Nature.  
- [huo2017riskbandit] Huo, H., & Fu, M. C. (2017). *Risk-aware multi-armed bandit and portfolio selection.* Royal Society Open Science, 4(1), 160641.  
- [fonseca2024banditnets] de Freitas Fonseca, P., et al. (2024). *Improving Portfolio Optimization Results with Bandit Networks.* arXiv:2410.04217.  
- [makridakis2019mmppo] Makridakis, J., et al. (2019). *Deep Reinforcement Learning for Cryptocurrency Market Making (A2C/PPO).* arXiv preprint.