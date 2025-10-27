# Project Timeline

**Project:** Reinforcement Learning for Cryptocurrency Portfolio Management  
**Team:** Jose Márquez Jaramillo & Taylor Hawks  
**Duration:** 6 weeks (Oct 27 - Dec 9, 2025)  
**Time Commitment:** 10-15 hours per week per person

---

## Overview

- **Code + Report Deadline:** December 2, 2025
- **Presentation:** December 9, 2025
- **Total Estimated Hours:** ~150 person-hours (75h each)
- **Agents:** LinUCB, DQN, REINFORCE (A2C dropped to manage scope)

---

## Week 1: Foundation (Oct 27 - Nov 3)

**Goal:** Data pipeline and environment setup  
**Total Hours:** 20-25h | Jose: 12-15h | Taylor: 8-10h

### Jose's Tasks (12-15h)
- [ ] Port existing data/indicator code to repo structure (2h)
- [ ] Implement `portfolio_env.py` - core MDP environment (6-8h)
  - State representation: (X_{t-L+1:t}, w_{t-1}, m_t)
  - Action space: portfolio weights on simplex
  - Step function: 7-day transitions
  - Reward: log-return - transaction costs
- [ ] Write environment unit tests (2-3h)
- [ ] Document environment API (1h)

**Deliverable:** Complete `data/` + `environment/` modules

### Taylor's Tasks (8-10h)
- [ ] Set up collaborative dev environment (3-4h)
  - Git repository structure
  - Colab notebook setup
  - Shared configuration
- [ ] Read LinUCB paper (Li et al. 2010) (2-3h)
- [ ] Explore LinUCB conceptually (2-3h)
  - Understand contextual bandit formulation
  - Sketch portfolio problem mapping
  - Identify key hyperparameters

**Deliverable:** Dev environment ready, LinUCB design document

**Weekend Checkpoint (Nov 2-3):** Test environment together on both machines

---

## Week 2: Baselines + LinUCB (Nov 4 - Nov 10)

**Goal:** Establish benchmarks and first RL agent  
**Total Hours:** 20-25h | Jose: 8-10h | Taylor: 12-15h

### Jose's Tasks (8-10h)
- [ ] Run existing benchmark code (EW, CapW, MVO) on test period (3-4h)
- [ ] Generate baseline results table (2-3h)
  - CAGR, Sharpe, max drawdown, turnover
- [ ] Read DQN papers (2-3h)
  - Mnih et al. 2015 (Nature DQN)
  - Lucarelli & Borrotti 2020 (crypto-specific)

**Deliverable:** Baseline performance numbers

### Taylor's Tasks (12-15h)
- [ ] Implement `linucb.py` (8-10h)
  - Context feature engineering from state
  - Ridge regression updates (A matrix, b vector)
  - UCB score calculation
  - Arm selection → portfolio weight mapping
  - Online learning loop
- [ ] Implement basic `train_loop.py` (2-3h)
- [ ] Train LinUCB on train set, validate (2h)

**Deliverable:** Working LinUCB agent with preliminary results

**Weekend Checkpoint (Nov 9-10):** Review LinUCB results together

---

## Week 3: DQN Implementation (Nov 11 - Nov 17)

**Goal:** Implement value-based RL agent  
**Total Hours:** 20-30h | Jose: 12-18h | Taylor: 8-12h

### Jose's Tasks (12-18h)
- [ ] Implement `replay_buffer.py` (2-3h)
- [ ] Implement `networks.py` - Q-network architecture (2-3h)
- [ ] Implement `dqn.py` (6-10h)
  - Discretize action space (e.g., ±5%, ±10% per asset)
  - ε-greedy exploration
  - Target network updates
  - TD loss and optimization
- [ ] Debug + initial training runs (2-3h)

**Deliverable:** DQN agent (basic version)

### Taylor's Tasks (8-12h)
- [ ] Read REINFORCE papers (2-3h)
  - Williams 1992
  - Jiang 2017 (crypto application)
- [ ] Start implementing `reinforce.py` (6-9h)
  - Policy network (softmax output)
  - Value baseline network
  - Monte Carlo return computation
  - Policy gradient loss

**Deliverable:** REINFORCE 50-70% complete

**Critical Decision Point (Nov 17):**
- ✅ DQN trains without crashes? → Continue
- ⚠️ Major issues? → Simplify or adjust scope

---

## Week 4: Complete Agents (Nov 18 - Nov 24)

**Goal:** Finalize all three RL agents  
**Total Hours:** 20-25h | Jose: 10-12h | Taylor: 10-13h

### Jose's Tasks (10-12h)
- [ ] Tune DQN hyperparameters (4-6h)
  - Learning rate, batch size, replay buffer size
  - ε decay schedule
  - Target network update frequency
- [ ] Run DQN training on full train set (2-3h)
- [ ] Validate DQN on validation set (2-3h)
- [ ] Start building `evaluator.py` (2-3h)

**Deliverable:** DQN validated and tuned

### Taylor's Tasks (10-13h)
- [ ] Complete `reinforce.py` implementation (4-6h)
- [ ] Debug and test REINFORCE (2-3h)
- [ ] Train REINFORCE on train set (2-3h)
- [ ] Validate on validation set (2-3h)

**Deliverable:** All 3 RL agents complete and validated

**Note:** Thanksgiving week (Nov 27-29) may have reduced productivity

---

## Week 5: Experiments + Evaluation (Nov 25 - Dec 1)

**Goal:** Run full experiments and generate results  
**Total Hours:** 25-30h | Jose: 15-18h | Taylor: 10-12h

### Jose's Tasks (15-18h)
- [ ] Implement complete `evaluator.py` with all metrics (4-5h)
- [ ] Run full experiments (8-10h compute + monitoring)
  - Each agent: 3 random seeds
  - Test period: 2023-2025
  - Collect all metrics (profitability, risk, efficiency)
- [ ] Generate figures and tables (3-4h)
  - Cumulative wealth curves
  - Sharpe ratio comparison
  - Turnover analysis

**Deliverable:** Complete experimental results + figures

### Taylor's Tasks (10-12h)
- [ ] Implement `visualization.py` (3-4h)
- [ ] Create supplementary analysis (3-4h)
  - Portfolio weight evolution
  - Per-asset allocation patterns
  - Trading frequency analysis
- [ ] Start report writing - Introduction + Related Work (4-5h)

**Deliverable:** Visualizations + report draft started

**Code Freeze (Nov 28):** No new features after this, only bug fixes

---

## Week 6: Report + Presentation (Dec 2 - Dec 9)

**Goal:** Complete deliverables  
**Total Hours:** 35-40h | Jose: 18-20h | Taylor: 17-20h

### Report Writing (Dec 2-4) - 25-30h total

**Jose's Sections (12-15h):**
- [ ] Methodology - Algorithms (3-4h)
- [ ] Experimental Setup (2-3h)
- [ ] Results section with tables/figures (4-5h)
- [ ] Discussion (2-3h)

**Taylor's Sections (13-15h):**
- [ ] Introduction + Motivation (3-4h)
- [ ] Related Work (3-4h)
- [ ] Methodology - MDP formulation (2-3h)
- [ ] Conclusion (2-3h)
- [ ] Final editing + references (3h)

**DEADLINE: December 2, 2025** ⚠️

### Presentation Prep (Dec 5-9) - 10-12h total

**Joint Tasks (5-6h each):**
- [ ] Extract key results from report
- [ ] Create presentation slides (use existing .tex template)
- [ ] Practice presentation timing (15-20 min)
- [ ] Rehearse together

**PRESENTATION: December 9, 2025** ⚠️

---

## Risk Mitigation

### High-Risk Items & Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Environment bug found late | 40% | High | Extensive unit tests Week 1 |
| DQN won't converge | 50% | Medium | Start simple, basic architecture |
| Hyperparameter tuning hell | 60% | Medium | Use defaults from papers |
| Report takes longer | 50% | Medium | Start intro/related work Week 5 |
| Week 6 needs >30h | High | High | Protect report time, freeze code Nov 28 |

### Backup Plans

**If behind by Nov 17:**
- Drop REINFORCE, focus on LinUCB + DQN only

**If behind by Nov 24:**
- Accept working implementations with limited tuning
- Document in report: "Future work: hyperparameter optimization"

**If Week 6 overwhelms:**
- Prioritize report (hard deadline Dec 2)
- Presentation can adapt from report sections

---

## Key Success Factors

1. ✅ **Start immediately** - Begin Week 1 tasks today (Oct 27)
2. ✅ **Weekly syncs** - 1-hour meeting every Sunday
3. ✅ **Daily standups** - Async: "What I did / What I'm doing / Blockers"
4. ✅ **Track hours** - Adjust plan if consistently over estimates
5. ✅ **Be brutally honest** - Cut scope early if falling behind
6. ✅ **Code freeze Nov 28** - Protect Week 6 for writing

---

## Notes

- **Algorithms dropped:** A2C (too complex for time available)
- **Rationale for 3 agents:** Covers diverse RL paradigms (bandit, value-based, policy gradient)
- **Time commitment:** 10-15 hours/week each is realistic for graduate coursework
- **Flexibility:** Timeline has buffer time for debugging and unexpected issues

---

**Last Updated:** October 27, 2025  
**Status:** Week 1 in progress

---

## Feedback

Please review this timeline and provide feedback on:
- [ ] Is the time allocation realistic?
- [ ] Are task assignments clear?
- [ ] Any concerns about specific weeks?
- [ ] Suggestions for adjustments?

Add comments below or create an issue for discussion.
