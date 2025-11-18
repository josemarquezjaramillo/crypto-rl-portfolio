# Weekly Project Update

**Date:** Tuesday, November 17, 2025  
**Name:** Jose Marquez Jaramillo  
**Teammate:** Taylor Hawks  
**Project:** Reinforcement Learning for Cryptocurrency Portfolio Management

---

## Changes in Project Objectives

No changes to project objectives this week. We remain on track with the original plan to implement a deep reinforcement learning framework for cryptocurrency portfolio management. Week 3 successfully delivered the DQN agent implementation as planned. However, a critical dataset bug was discovered during smoke testing (DST timestamp mismatch causing NaN forward returns), which required immediate investigation and fixing. This unplanned debugging effort consumed approximately 6-8 hours but was essential for ensuring data quality across all future training runs.

---

## Your Tasks & Accomplishments

This week I completed **Week 3: DQN Implementation**. The primary focus was implementing a Deep Q-Network agent for cryptocurrency portfolio management using a discrete action catalog approach. The DQN treats portfolio allocation as a discrete action space problem by defining a catalog of 47 feasible portfolio strategies and learning Q-values to select the optimal strategy given the current market state. During initial smoke testing, I discovered a critical dataset bug that caused NaN values in training, which led to a deep debugging session and ultimately a complete dataset re-export with a timestamp normalization fix.

1. **DQN Architecture & Action Catalog (8-10h)**: Following the approach outlined in Lucarelli & Borrotti (2020), I designed a discrete action catalog of 47 portfolio strategies to make the continuous allocation problem tractable for Q-learning. The catalog is organized into three categories: (1) Equal-weight strategies across top-K assets for K ∈ {2, 3, 5, 10, 15, 20, 30}, which provides diversified baseline allocations; (2) Sparse allocations including 1-asset, 2-asset, and 3-asset portfolios with varying concentration levels (e.g., 50/50, 70/30, 90/10), which allows the agent to make concentrated bets; and (3) Diversified allocations with 3-5 assets and balanced weights, which provides middle-ground risk profiles. Each strategy is implemented as a callable function that takes the current observation, asset IDs, and previous weights, then returns a valid weight vector (sum=1, non-negative). The catalog handles the variable universe size A_t dynamically, which is critical since the number of tradable assets changes daily based on monthly index membership and cold-start eligibility rules. I implemented the action catalog in `agents/dqn/action_catalog.py` (158 lines) and validated that all 47 strategies produce valid simplex allocations on real market data.

    For the neural network architecture, I designed a StateEncoder that handles the variable-size observation tensor X_t ∈ ℝ^{A_t × 4 × 60} by using average pooling over the asset dimension, producing a fixed 256-dimensional state embedding regardless of how many assets are tradable on a given day. This approach avoids the padding overhead of fixed-size architectures while maintaining a consistent representation for the Q-network. The QNetwork is a standard 3-layer MLP that takes the concatenated state embedding (256-dim) and previous portfolio weights (padded to 50 assets) as input, and outputs Q-values for each of the 47 catalog actions. The total network has approximately 104,000 parameters—relatively shallow compared to image-based DQN architectures, which aligns with Lucarelli & Borrotti's finding that financial features don't require deep networks. Both networks support proper device placement via `.to(device)` methods for CPU/GPU compatibility, which became critical during debugging (more on that below).

2. **Training Infrastructure & Experience Replay (4-6h)**: I implemented the core DQN training loop in `agents/dqn/dqn_agent.py` (462 lines), which extends our `BaseAgent` infrastructure from Week 1. The agent uses standard DQN components: an experience replay buffer with 10,000 capacity for breaking temporal correlations in the financial time series, a separate target Q-network updated every 100 steps to prevent moving target issues during TD learning, epsilon-greedy exploration with decay from 1.0 to 0.1 over training episodes, and Huber loss for robustness to reward outliers. The implementation integrates cleanly with our existing infrastructure—it uses the `MetricsTracker` for episode logging, implements the required abstract methods (`select_action`, `update`, `save`, `load`), and respects all portfolio constraints through the environment interface (long-only, fully-invested, turnover cap τ=0.30). The agent configuration uses Adam optimizer with learning rate 1e-4, batch size 64, and discount factor γ=0.99, following the hyperparameter ranges suggested in the DQN literature for financial applications.

3. **Critical Dataset Bug Investigation & Fix (6-8h)**: During initial smoke testing of the DQN agent (running 5 training episodes), I encountered NaN values in rewards and portfolio values starting at step 64 (date 2018-11-04), then recurring approximately every 180 steps. This was alarming because NaN propagation quickly destabilizes neural network training. I conducted a systematic debugging investigation, starting with the hypothesis that the DQN implementation had a bug. I created diagnostic scripts to validate the action catalog (all 47 strategies produced valid weights), the Q-network forward/backward passes (gradients were finite), and the environment reward computation (manual calculations matched expected values). After ruling out the DQN and environment code, I discovered that the root cause was in the dataset itself: the forward returns for 2018-11-04 were all NaN.

    Digging deeper into the data pipeline, I found that the database stores timestamps with hour precision (e.g., 2018-11-04 04:00:00) and that US Daylight Saving Time ended on 2018-11-04, causing timestamps to shift from 04:00 UTC to 05:00 UTC starting 2018-11-05. When `data_builder.py` used `pd.date_range(..., freq='D')` to create a uniform daily index, it generated timestamps at 04:00:00 for all dates. The subsequent reindex operation looked for exact timestamp matches, so `2018-11-05 04:00:00` (from date_range) did not match `2018-11-05 05:00:00` (from the database), causing all data from 2018-11-05 onward to become NaN. When forward returns were computed as (close_t+1 - close_t) / close_t, the NaN in close_t+1 propagated to the forward returns, and the environment received NaN rewards.

    The fix was elegantly simple: I added `raw["timestamp"] = raw["timestamp"].dt.normalize()` in `data_builder.py` to strip the time component from all timestamps before pivoting and reindexing. This ensures all timestamps are normalized to midnight (00:00:00), making the reindex operation robust to DST shifts or any other hour-level inconsistencies in the database. I re-exported the entire dataset using `python -m data.data_exporter`, which took approximately 20 minutes. Verification showed that the 2018-11-04 forward returns changed from `[nan nan nan ...]` to valid returns `[-0.0058, 0.0048, 0.0417, ...]`, and a full dataset scan confirmed 0 NaN values across all 1,948 days and 19,480 forward returns. This fix not only resolved the immediate training issue but also improved the robustness of our data pipeline for any future database timestamp inconsistencies.

**Difficulties encountered and overcome:**

The biggest challenge this week was debugging the dataset NaN issue, which required tracing through multiple layers of the system (DQN agent → environment → dataset backend → data builder → database) to find the root cause. The DST timestamp mismatch was particularly insidious because it only manifested on specific dates (DST boundaries) and the error message (NaN in rewards) gave no indication of the underlying timestamp problem. I learned the value of systematic debugging: instead of making random fixes, I methodically validated each component (action catalog, Q-network, environment, dataset) until I isolated the failure point. Creating small diagnostic scripts to test each component independently was crucial—it allowed me to definitively rule out the DQN implementation and focus on the data pipeline.

Another technical challenge was handling the variable universe size A_t in the DQN architecture. Standard DQN implementations assume fixed action spaces, but our portfolio problem has A_t varying from 10-30 assets depending on the day. Using average pooling in the StateEncoder to produce a fixed 256-dim embedding was the key insight—it avoids padding overhead while maintaining a consistent representation. I also encountered a device placement bug where the StateEncoder's projection layer wasn't being moved to CUDA, causing "tensors on different devices" errors during training. The fix was implementing a proper `.to(device)` method on StateEncoder that recursively moves all internal layers.

**What I learned this week:**

The most important technical lesson from this week is the critical importance of **timestamp normalization in financial data pipelines**. Financial databases often store timestamps with varying precision (hourly, minute-level) depending on the data source, and phenomena like DST create subtle mismatches that cause data loss during reindexing. The lesson is to always normalize timestamps to a consistent precision (in our case, date-only at midnight) as early as possible in the pipeline, before any pivoting or reindexing operations. This defensive programming practice prevents an entire class of data quality bugs.

I also gained a deeper appreciation for **discrete action space design in continuous control problems**. The DQN literature typically deals with naturally discrete problems (Atari games with button presses), but portfolio allocation is fundamentally continuous (weights on the simplex). Lucarelli & Borrotti's insight—defining a fixed catalog of template portfolios rather than discretizing each asset's weight independently—elegantly solves the curse of dimensionality. With A=20 assets and 10 weight levels per asset, independent discretization gives 10^20 actions (intractable), but a fixed 47-strategy catalog remains manageable while still covering diverse allocation behaviors (concentrated, balanced, diversified). The key is ensuring the catalog "spans" the interesting regions of the portfolio simplex.

Finally, I learned the value of **comprehensive smoke testing before full training runs**. The smoke test suite I created (`agents/dqn/smoke_test.py`) runs 5 short episodes, validates action catalog outputs, checks for NaN in TD loss and Q-values, and tests checkpoint save/load—all in under 5 minutes. This caught the dataset bug immediately, saving potentially hours of wasted training time and GPU compute. The investment in smoke testing infrastructure (2-3 hours to write) paid for itself many times over during debugging.

---

## Teammate's Tasks & Accomplishments

Based on my understanding, Taylor is scheduled to begin work on the **Week 4: REINFORCE+baseline implementation** starting November 18, 2025. As of this update (November 17), Week 4 has not yet started, so Taylor's work is pending. He will implement a policy gradient agent that extends our `BaseAgent` infrastructure and leverages the re-exported clean dataset from this week's DST bug fix. We have coordinated that Taylor will use a continuous action space approach (softmax over asset weights) rather than the discrete catalog used by DQN, which will allow us to compare these two action space designs in the final Week 5 evaluation. I've shared the smoke testing framework and dataset verification procedures with Taylor to help him avoid similar data quality issues during his implementation.

---

## GitHub Activity

**Repository:** https://github.com/josemarquezjaramillo/crypto-rl-portfolio

**Completed this week:**
- ✅ DQN agent implementation (`agents/dqn/`)
  - `action_catalog.py` - 47 portfolio strategies
  - `networks.py` - StateEncoder + QNetwork architectures
  - `replay_buffer.py` - Experience replay buffer
  - `dqn_agent.py` - Main DQN agent with ε-greedy, target networks
  - `smoke_test.py` - Comprehensive end-to-end validation
- ✅ Critical dataset bug fix
  - `data/data_builder.py` - Timestamp normalization via `.dt.normalize()`
  - `data/data_exporter.py` - Fixed import paths
  - `dataset_v1/` - Complete re-export (0 NaN values verified)
- ✅ Documentation updates
  - `documentation/PROJECT_SPECIFICATION.md` - Added DQN implementation details, DST fix
  - `WEEK3_PROJECT_UPDATE.md` - This document

**Milestone progress:** Week 3: DQN Implementation (complete)

---

## Risks, Concerns & Timeline Status

**Risks and concerns:**

The main risk identified this week is the potential for similar timestamp-related data quality issues in other date ranges beyond the DST boundary on 2018-11-04. While the timestamp normalization fix should prevent future DST-related bugs, there could be other database anomalies (missing days, duplicate timestamps, timezone inconsistencies) that haven't manifested yet. Before Week 5 full training runs, I should conduct a comprehensive data quality audit—checking for gaps in the daily index, verifying that all forward returns are finite, and ensuring no other NaN patterns exist in the dataset. This audit would take approximately 2-3 hours but could save significant debugging time later.

Another concern is the DQN action catalog design. While the 47-strategy catalog is diverse and tractable, I won't know if it's sufficiently expressive until I run full training in Week 5. If the DQN performance is significantly worse than the continuous-action REINFORCE agent, it might indicate that the discrete catalog is too restrictive. However, this is an inherent trade-off in discretizing continuous control problems, and Lucarelli & Borrotti's results suggest that a well-designed catalog can achieve competitive performance.

**On track to finish on time:**

Yes. Week 3 is complete with the DQN implementation validated and the dataset bug fixed. The DST debugging took longer than expected (6-8 hours unplanned), but it was a necessary investment in data quality that will benefit all future training runs. Taylor's Week 4 REINFORCE implementation starts tomorrow (November 18), and we're on track for Week 5 joint evaluation and final report. The timeline remains feasible.

---

## Key References

This week's work drew primarily from the following papers:

- **Mnih et al. (2015)** - "Human-level control through deep reinforcement learning" - Core DQN architecture including experience replay, target networks, and epsilon-greedy exploration. Foundational reference for understanding how to stabilize Q-learning with neural networks. [[PDF](https://github.com/josemarquezjaramillo/crypto-rl-portfolio/blob/main/reference_papers/mnih2015dqn%20-%20Human-level%20control%20through%20deep%20reinforcement%20learning.pdf)]

- **Lucarelli & Borrotti (2020)** - "Deep Reinforcement Learning for Cryptocurrency Trading" - Crypto-specific DQN adaptations including discrete action catalog design, regime-aware validation sampling, shallow network architectures for financial features, and transaction cost modeling. Most directly applicable paper for portfolio DQN. [[PDF](https://github.com/josemarquezjaramillo/crypto-rl-portfolio/blob/main/reference_papers/lucarelli2020dqlcrypto%20-%20Deep%20Reinforcement%20Learning%20for%20Cryptocurrency%20Trading.pdf)]

- **Jiang et al. (2017)** - "A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem" - Referenced for the Portfolio Vector Memory (PVM) mechanism (conditioning on previous weights w_{t-1}), which helps agents learn to minimize turnover costs. While this is a policy-gradient paper, the PVM concept applies to DQN as well. [[PDF](https://github.com/josemarquezjaramillo/crypto-rl-portfolio/blob/main/reference_papers/jiang2017eIIE%20-%20A%20Deep%20Reinforcement%20Learning%20Framework%20for%20the%20Financial%20Portfolio%20Management%20Problem.pdf)]

- **Jiang et al. (2016)** - "Cryptocurrency Portfolio Management with Deep Reinforcement Learning" - Consulted for understanding data quality issues in crypto RL (survivorship bias, gap filling, regime nonstationarity). Informed the debugging process for the DST timestamp bug. [[PDF](https://github.com/josemarquezjaramillo/crypto-rl-portfolio/blob/main/reference_papers/jiang2016drlt%20-%20Cryptocurrency%20Portfolio%20Management%20with%20Deep%20Reinforcement%20Learning..pdf)]
