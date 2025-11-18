# Weekly Project Update

**Date:** Tuesday, November 11, 2025  
**Name:** Jose Marquez Jaramillo  
**Teammate:** Taylor Hawks  
**Project:** Reinforcement Learning for Cryptocurrency Portfolio Management

---

## Changes in Project Objectives

No changes to project objectives this week. We remain on track with the original plan to implement a deep reinforcement learning framework for cryptocurrency portfolio management. This week's focus shifted slightly from running baseline benchmarks to prioritizing DQN research and preparation, as baseline evaluation can be deferred to the final analysis phase (Week 5).

---

## Your Tasks & Accomplishments

This week I completed **Week 2: Baselines + Research**. The primary focus was on researching DQN implementation details and understanding how to adapt deep Q-learning to the portfolio management problem. While I had planned to run baseline benchmarks (Equal Weight, Cap Weight, MVO) on the test period, I decided to defer this to Week 5 evaluation, as the immediate priority is getting the RL agents (DQN, REINFORCE, LinUCB) implemented and trained.

1. **DQN Paper Reading & Research (8-10h)**: I conducted a review of the foundational DQN literature and crypto-specific adaptations:

    - **Mnih et al. (2015) - "Human-level control through deep reinforcement learning"**: Studied the core DQN architecture including experience replay, target networks, and frame stacking. Key takeaways relevant to our portfolio problem:
        - **Experience Replay**: Critical for breaking temporal correlations in financial time series. Our portfolio environment exhibits strong autocorrelation, so replay buffer will be essential for stable learning.
        - **Target Networks**: Using a separate target Q-network (updated every C steps) prevents moving target problem during TD learning. This is particularly important for portfolio problems where market regimes can shift.
        - **Epsilon-Greedy Exploration**: Need to carefully tune epsilon decay for financial markets—too much exploration can be costly, but too little prevents discovering better portfolio allocations.

    - **Lucarelli & Borrotti (2020) - "Deep Reinforcement Learning for Cryptocurrency Trading"**: This paper provides the most directly applicable insights for our crypto portfolio DQN:
        - **Action Space Discretization**: They discretize continuous portfolio weights into a finite action catalog (e.g., "increase BTC by 10%", "decrease ETH by 5%"). Our PROJECT_SPECIFICATION.md already adopts this approach—defining a catalog of feasible portfolios (equal-weight top K, sparse 2-3 asset mixes, etc.) that the DQN selects from.
        - **State Representation**: They use 30-60 day lookback windows with normalized OHLCV features, which matches our 60-day windows. Confirmed that price normalization (dividing by current close) is essential for stationarity.
        - **Train/Val/Test Split Strategy**: They use **regime-specific validation sampling** rather than a single contiguous validation period. This directly informed our PROJECT_SPECIFICATION.md approach of defining multiple ~20-day validation windows across different crypto regimes (crash, bull, deleverage, chop). This prevents hyperparameter overfitting to a single market regime.
        - **Transaction Costs**: They model proportional costs as `cost = c * ||Δw||₁`, which we've already implemented in our environment. Their experiments show that ignoring costs leads to pathological over-trading.
        - **Network Architecture**: They use relatively shallow networks (2-3 dense layers, 128-256 units) rather than deep CNNs, since financial features are already engineered. This suggests our DQN doesn't need to be overly complex.

    - **Jiang et al. (2017) - EIIE Framework**: While this is a policy-gradient paper, their discussion of the Portfolio Vector Memory (PVM) mechanism is crucial for DQN too. Conditioning on previous weights `w_{t-1}` helps the agent learn to minimize turnover costs. Our environment already includes this in the observation space.

2. **Baseline Agent Implementation (External Work)**: I have implemented baseline agents (Equal Weight, Market Cap Weight, Historical Mean-Variance Optimization) in the separate [kallos_portfolios](https://github.com/josemarquezjaramillo/kallos_portfolios/) repository. This repository includes:

    - Three baseline strategies with identical optimization procedures for fair comparison
    - Comprehensive evaluation framework with CAGR, Sharpe ratio, max drawdown, turnover, and statistical hypothesis testing
    - VectorBT-based backtesting with realistic transaction costs
    - QuantStats integration for tearsheet generation

    **Status**: The code is complete and functional, but I have **not yet integrated it into the crypto-rl-portfolio repository**. Since our immediate focus is on implementing the RL agents (DQN, REINFORCE, LinUCB), I've decided to defer running the baseline benchmarks until Week 5 (Experiments + Evaluation). At that point, I can run all baselines and RL agents together for comprehensive comparison. This prevents premature optimization and keeps us focused on the core RL implementation.

3. **DQN Implementation Planning**: Based on the paper review, I've outlined the DQN architecture for Week 3 implementation:

    - **State Processing**: The environment already provides `(X_t, w_{t-1})` where `X_t ∈ ℝ^{A_t × 4 × 60}`. I'll flatten or use a 1D CNN over the temporal dimension, then concatenate with `w_{t-1}`.
    - **Action Catalog**: Define ~20-50 candidate portfolios (equal-weight top-K, sparse allocations, diversified mixes). Each catalog entry becomes a discrete action for the DQN.
    - **Replay Buffer**: Standard circular buffer storing `(s_t, a_t, r_t, s_{t+1}, done)`. Buffer size ~10,000-50,000 transitions (will tune).
    - **Target Network**: Update every 1,000 steps (will tune based on validation performance).
    - **Exploration**: Epsilon-greedy with decay from 1.0 → 0.1 over ~50% of training episodes.
    - **Network**: 2-3 dense layers (256-512 units), ReLU activations, outputting Q-values for each catalog action.

**Difficulties encountered and overcome:**

The main challenge this week was deciding how to handle the baseline implementations. Initially, I planned to run full benchmark experiments this week, but after reflection, I realized this would be premature. The baseline results are only meaningful when compared against the RL agents, which don't exist yet. Running benchmarks now would require re-running them later anyway (to ensure consistency in evaluation setup). By deferring to Week 5, I can focus entirely on getting the RL implementations working, which is the core contribution of this project.

Another consideration was understanding how to discretize the continuous portfolio weight space for DQN. The Lucarelli paper provided the key insight: instead of discretizing each asset's weight independently (which leads to combinatorial explosion), define a small catalog of sensible portfolio templates. This keeps the action space tractable (~20-50 actions) while still covering diverse allocation strategies.

**What I learned this week:**

The most important lesson from the DQN literature is the critical role of **regime-aware validation** in financial RL. Traditional ML often uses a single held-out validation set, but crypto markets exhibit such strong regime nonstationarity that this leads to overfitting. Lucarelli & Borrotti's approach of sampling validation windows from different market regimes (2020 COVID crash, 2021 bull run, 2022 deleveraging, 2023 sideways chop) directly addresses this. Our PROJECT_SPECIFICATION.md already implements this philosophy, which gives me confidence we're following best practices.

I also learned that **shallower networks often work better for portfolio problems** than deep architectures. This makes sense: unlike image or text data where deep networks extract hierarchical features, our OHLCV features are already engineered. A 2-3 layer network with proper regularization (dropout, L2) is usually sufficient. Overcomplicating the architecture risks overfitting to training data.

Finally, I gained a deeper appreciation for the **action space design trade-off** in portfolio DQN. Discretizing each asset's weight independently (e.g., {0%, 10%, 20%, ..., 100%} per asset) leads to `11^A` possible actions for A assets—intractable. But defining a fixed catalog of ~50 template portfolios reduces the action space dramatically while still allowing flexible allocations. The key is ensuring the catalog covers the "corners" of the feasible region (concentrated vs diversified, momentum vs mean-reversion, etc.).

---

## Teammate's Tasks & Accomplishments

Taylor has made significant progress on the REINFORCE+baseline (policy gradient) agent implementation this week:

1. **Policy Gradient Agent Implementation**: Taylor successfully implemented the core policy gradient agent with the following components:
   
   - **Forward Propagation**: Completed the neural network forward pass for the policy gradient network, which takes the environment observation and outputs portfolio allocation probabilities.
   
   - **Backward Propagation**: Implemented the backpropagation algorithm for computing policy gradients. This includes proper gradient computation for the policy loss function.
   
   - **Training Loop Integration**: Built the complete training pipeline that integrates with our `BaseAgent` infrastructure, including:
     - Episode-based data collection
     - Gradient accumulation across episodes
     - Optimizer steps with proper zeroing of gradients
     - Loss tracking and logging
   
   The implementation successfully runs training episodes and produces meaningful reward signals, as evidenced by his working code demonstrations.

2. **BaseAgent Design Discussion**: Taylor raised important questions about the `BaseAgent` architecture, specifically regarding whether `save()` and `load()` methods should be abstract or implemented in the parent class. We've agreed to discuss this design decision in our next meeting on Tuesday after lecture. This shows thoughtful consideration of code architecture and reusability.

3. **Technical Challenges & Solutions**: Taylor identified and resolved confusion around deterministic vs. non-deterministic policy gradient behavior during action selection. This is a critical distinction for training (where stochasticity enables exploration) versus evaluation (where deterministic behavior may be preferred for consistent performance measurement).

**Next steps for Taylor**: Continue refining the REINFORCE+baseline implementation, conduct initial training runs to validate the agent's learning behavior, and collaborate on finalizing the `BaseAgent` design patterns for consistency across all agent implementations (DQN, LinUCB, REINFORCE).

---

## GitHub Activity

**Repository:** https://github.com/josemarquezjaramillo/crypto-rl-portfolio

**Completed this week:**
- ✅ DQN literature review and architecture planning
- ✅ Baseline agent implementation (external repo: [kallos_portfolios](https://github.com/josemarquezjaramillo/kallos_portfolios/))
- ⏸️ Baseline benchmarking deferred to Week 5 evaluation phase

**In Progress:**
- 🔄 DQN implementation planning and preparation for Week 3

**Milestone progress:** Week 2: Baselines + Research (completed with scope adjustment)

---

## Risks, Concerns & Timeline Status

**Risks and concerns:**

The main risk this week is ensuring we stay on schedule for DQN implementation in Week 3. By deferring baseline benchmarking, I've freed up time to focus on the DQN agent, but this means Week 3 will be critical. If DQN implementation takes longer than expected (10-12h estimated), we may need to simplify the architecture or reduce hyperparameter search.

Another concern is the **action catalog design**. While the literature suggests 20-50 template portfolios is tractable, I need to ensure the catalog is diverse enough to find good solutions while remaining computationally feasible. This will likely require some iteration during early DQN training.

**On track to finish on time:**

Yes. By deferring baseline evaluation to Week 5, I've actually reduced scope creep and can focus on the core RL implementations. The DQN paper review this week has prepared me well for next week's implementation. I'm confident we can complete DQN in Week 3 as planned.

---

## Key References

This week's work drew primarily from the following papers:

- **Mnih et al. (2015)** - "Human-level control through deep reinforcement learning" - Core DQN architecture: experience replay, target networks, epsilon-greedy exploration.

- **Lucarelli & Borrotti (2020)** - "Deep Reinforcement Learning for Cryptocurrency Trading" - Crypto-specific DQN adaptations: action space discretization via portfolio catalog, regime-aware validation sampling, transaction cost modeling, shallow network architectures for financial features.

- **Jiang et al. (2017)** - "A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem" - Portfolio Vector Memory (PVM) mechanism for incorporating previous weights `w_{t-1}` into state representation, critical for learning to minimize turnover costs. 

- **Jiang et al. (2016)** - "Cryptocurrency Portfolio Management with Deep Reinforcement Learning" - Early crypto portfolio RL work emphasizing regime nonstationarity and validation across multiple market periods. 
