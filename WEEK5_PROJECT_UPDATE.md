# Weekly Project Update

**Date:** Tuesday, December 2, 2025  
**Name:** Jose Marquez Jaramillo  
**Teammate:** Taylor Hawks  
**Project:** Reinforcement Learning for Cryptocurrency Portfolio Management

---

## Changes in Project Objectives

We replaced **LinUCB with Double DQN (DDQN)** due to time constraints. Our final agent lineup is:

1. **DQN** - Discrete action catalog (70 portfolio strategies)
2. **DDQN** - Double Q-learning to reduce overestimation bias
3. **REINFORCE** - Policy gradient with continuous action space

This enables direct comparison of value-based (DQN/DDQN) vs. policy-gradient (REINFORCE) approaches.

---

## Your Tasks & Accomplishments

This week I completed **Week 5: Experiments + Evaluation**, focusing on publication-ready evaluation infrastructure and comprehensive agent comparisons.

### 1. Evaluation Module Reorganization (8-10h)

Gap analysis against Jiang (2017) and Lucarelli (2020) revealed missing visualizations. Created:

- `evaluation/metrics.py` - CAGR, Sharpe, Sortino, Calmar, max drawdown, volatility, hit rate
- `evaluation/tables.py` - LaTeX/Markdown/HTML generators with best-metric highlighting
- `evaluation/run_full_evaluation.py` - CLI with `--detailed` flag for time-series charts
- `DetailedAgentResult` dataclass for per-step data collection (weights, returns, turnovers)

### 2. New Visualization Functions (6-8h)

Added 5 functions to `visualizer.py`: cumulative returns with drawdown subplot, rolling Sharpe, daily returns distribution, weight evolution, and multi-agent allocation comparison.

### 3. Code Cleanup (2h)

Removed 3 redundant files from `baselines/` (~1,552 lines). All functionality consolidated into `evaluation/` module.

### 4. Full Evaluation Run

Test set evaluation (2024-01-01 to 2025-10-31, 646 days):

| Agent | Return (%) | CAGR (%) | Sharpe | Sortino | Max DD (%) | Turnover (%) |
|-------|------------|----------|--------|---------|------------|--------------|
| **Mean-Variance** | **366.5** | **139.1** | **1.64** | **1.69** | **40.0** | 13.5 |
| Equal Weight | 168.5 | 74.9 | 1.22 | 1.25 | 48.9 | 0.3 |
| Market Cap | 159.3 | 71.5 | 1.24 | 1.30 | 44.2 | 0.3 |
| DQN | 155.5 | 70.0 | 1.15 | 1.18 | 52.6 | 10.9 |
| DDQN | 144.5 | 65.9 | 1.14 | 1.16 | 52.6 | 8.4 |

### 5. Key Observations

- **RL agents underperform baselines**: DQN/DDQN achieve lower returns and Sharpe ratios than all baselines.
- **Mean-Variance dominates**: 366% return vs. ~155% for DQN.
- **Higher turnover without alpha**: DQN/DDQN trade more but don't capture additional returns.

Possible explanations: discrete action catalog is too restrictive; 2024-2025 bull market favors momentum (which Mean-Variance captures); regime shift from training (2018-2023) to test period.

**Key lesson**: RL agents don't automatically outperform simple baselines in finance—consistent with Lucarelli & Borrotti (2020) findings on regime sensitivity.

---

## Teammate's Tasks & Accomplishments

Taylor has been working on the **REINFORCE+baseline implementation**:

- Policy gradient agent with continuous action space (softmax over weights)
- Value function baseline for variance reduction
- Initial training runs on development set

**Status**: REINFORCE ready for integration into final comparison.

He has also adjusted his agent implementation to easily incorporate into our base agent infrastructure and so that we can use the evaluator code that I implemented for all three agents. 

---

## GitHub Activity

**Repository:** https://github.com/josemarquezjaramillo/crypto-rl-portfolio

**Completed:**
- ✅ Evaluation module reorganization
- ✅ 5 new visualization functions
- ✅ Full test set evaluation with LaTeX/Markdown tables
- ✅ Code cleanup (~1,552 lines removed)

**Milestone:** Week 5: Experiments + Evaluation (in progress)

---

## Risks, Concerns & Timeline Status

**Main concern**: RL underperformance relative to baselines. For the final report, we will analyze per-regime performance and discuss discrete vs. continuous action space limitations.

**On track**: Yes. Remaining tasks: integrate REINFORCE results, run validation set analysis, write final report.

---

## Key References

- **Jiang et al. (2017)** - Visualization standards and evaluation metrics
- **Lucarelli & Borrotti (2020)** - DQN evaluation methodology, regime-aware validation
- **Mnih et al. (2015)** - Double DQN implementation
