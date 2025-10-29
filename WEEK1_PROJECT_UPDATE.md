# Weekly Project Update

**Date:** Tuesday, October 28, 2025  
**Name:** Jose Marquez  
**Teammate:** [Partner's Name]  
**Project:** Reinforcement Learning for Cryptocurrency Portfolio Management

---

## Changes in Project Objectives

No changes to project objectives this week. We remain on track with the original plan to implement a deep reinforcement learning framework for cryptocurrency portfolio management based on Jiang et al. (2017).

---

## Your Tasks & Accomplishments

This week, I completed **Week 1: Data Infrastructure** - all three core data modules are now implemented, tested, and documented. I started by implementing `data_loader.py` and `data_processor.py` to handle database queries and tensor transformations. After extensive research into the Jiang et al. (2017) paper and their GitHub implementation, I made several critical design decisions that required iteration and refinement.

The most substantial work was implementing `data_transformer.py` (623 lines), which orchestrates the full data pipeline. Before implementation, I conducted thorough research to answer 6 key architectural questions: (1) how to handle monthly index transitions, (2) rolling window strategy, (3) variable vs fixed coin counts, (4) train/test split methodology, (5) where to handle transaction costs, and (6) caching strategy. Through careful analysis of Jiang's actual GitHub implementation, I discovered they used fixed coin counts and continuous rolling windows - insights that significantly shaped our approach.

I implemented a hybrid backfilling strategy (forward fill ≤2 days, interpolate 3-7 days, drop >7 days) to maintain data quality while maximizing usable data. The pipeline supports extended rolling windows adapted to monthly boundaries, creating ~2,640 training samples from ~88 months of data. I also implemented NPZ caching to enable fast iteration (~5-10 seconds vs 10-30 minutes) and easy sharing with my partner via Google Drive. After fixing import path issues and pandas deprecation warnings, all unit tests now pass successfully.

**Difficulties encountered and overcome:**

The main challenge was determining the optimal train/validation/test split strategy. We iterated through three different approaches: initially proposed 2019-2021/2022/2023-2025, then refined to 2019-2022/2023/2024-2025 for better balance, and finally settled on Jul 2018-2022/2023/2024-2025 to maximize training data using reliable data from July 2018 onwards. Another technical challenge was resolving Python import errors between modules - the initial implementation used absolute imports (`from config import`) which failed when running tests. I fixed this by using proper relative imports (`from data.config import`) and updating all three modules consistently. I also encountered pandas FutureWarnings for deprecated methods and resolved them by switching from `fillna(method='ffill')` to `ffill()` and adding `infer_objects(copy=False)` before interpolation.

**What I learned this week:**

This week reinforced the critical importance of reading original paper implementations, not just the papers themselves. By consulting Jiang's GitHub repository, I discovered implementation details (fixed coin counts, exact window handling) that weren't fully specified in the paper. I also learned the value of extensive design discussion before coding - answering those 6 architectural questions upfront saved significant refactoring time. The experience of balancing "simplicity and effectiveness over over-engineered complexity" taught me that pragmatic solutions (like hybrid backfilling) often work better than theoretically perfect but complex approaches. Finally, I gained deeper appreciation for proper Python package structure and the importance of using relative imports in multi-module projects.

---

## Teammate's Tasks & Accomplishments

Based on my understanding, [Teammate name] will be working on:

[This section to be filled in coordination with partner - they are working on the RL environment and algorithm implementation side while I handle data infrastructure.]

---

## GitHub Activity

**Repository:** https://github.com/josemarquezjaramillo/crypto-rl-portfolio

**Completed this week:**
- ✅ data/config.py - Configuration management (~90 lines)
- ✅ data/data_loader.py - Database queries (~145 lines)
- ✅ data/data_processor.py - Tensor transformations (~195 lines)
- ✅ data/data_transformer.py - Orchestration with caching (~623 lines)
- ✅ tests/test_data_transformer.py - Unit tests (3/3 passing)
- ✅ Documentation: README.md, IMPLEMENTATION_SUMMARY.md (~750 lines)
- ✅ Cache infrastructure: cache/README.md, .gitignore updates

**Milestone progress:** Week 1: Data Infrastructure - **100% complete (7 of 7 tasks)**

---

## Risks, Concerns & Timeline Status

**Risks and concerns:**

[Brief paragraph describing any blockers, technical risks, or concerns. If none, write: "No significant risks at this time."]

**On track to finish on time:** [YES / NO]

[Optional: 1 sentence explanation if needed]
