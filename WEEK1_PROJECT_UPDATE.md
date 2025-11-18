# Weekly Project Update

**Date:** Tuesday, November 4, 2025  
**Name:** Jose Marquez Jaramillo 
**Teammate:** Taylor Hawks  
**Project:** Reinforcement Learning for Cryptocurrency Portfolio Management

---

## Changes in Project Objectives

No changes to project objectives this week. We remain on track with the original plan to implement a deep reinforcement learning framework for cryptocurrency portfolio management based on Jiang et al. (2017). We have made a small adjustment on the definition of the environment and state space. In order to keep similar cadence to the references we are using, I have decided to change the rebalancing period from weekly to daily. 

---

## Your Tasks & Accomplishments

This week I completed **Week 1: Data Infrastructure**. That entailed creating a data pipeline that would generate datasets that we could use for all of our agents. Because of this, I had to revisit the design of our state space definitions as well as consider data processing techniques. This also entailed the creation of a base environment that we could use to train our agents. Considering that our more complex agents so far are based on DQN and REINFORCE+baseline, we have followed carefully the implementations of Jiang et al. (2017) and Lucarelli & Borrotti (2019).

1. **Dataset generation**: The objective here was to create development and testing datasets that could be used interchangeably across the different agents. The entire specification of what we ended up implementing can be found in our [PROJECT_SPECIFICATION.md](https://github.com/josemarquezjaramillo/crypto-rl-portfolio/blob/main/documentation/PROJECT_SPECIFICATION.md) file. On route to generating the different datasets; exporting the datasets; and loading into memory the datasets; from my local cryptocurrency database, I generated a series of files that are located in the [data](https://github.com/josemarquezjaramillo/crypto-rl-portfolio/tree/main/data) directory of the repository. These files are:

    - `data_loader.py`: Read the data from the different tables in my current cryptocurrency database.

    - `data_builder.py`: Builds tensors using normalization and the data provided from `data_loader.py`.

    - `data_exporter.py`: Generates a series of files that can be exported, including some metadata relevant to the dataset.

    - `dataset_loader.py`: Can take a directory path and read all files into memory. This was mainly created to be able to generate agents individually while using the same data, state specification, and parameters. 



2. **Environment Generation**: We needed to generate a baseline environment that would implement a Markov Decision Process for our agents. I implemented an environment located in the [environment](https://github.com/josemarquezjaramillo/crypto-rl-portfolio/tree/main/environment) directory of the repository. The environment implements  daily rebalancing, computes state action rewards (including transaction costs), and also implements some basic trading constraints. For this I created a few files:

    - `dataset_backend.py`: provides a concrete implementation of the DataBackend interface expected by PortfolioEnv in `environment.py`, wrapping the ExportedDataset class from `dataset_loader.py`. It is basically a bridge. 

    - `environment.py`: This module implements a Markov Decision Process (MDP) for portfolio management with daily rebalancing, transaction costs, and realistic trading constraints as specified in PROJECT_SPECIFICATION.md.

    - `environment_smoke_run.py`: Implements an entire demonstration of the use of the environment. It brings together the use of the datasets, creates a backend adapter, intializes the environment, runs training episodes, and produces some basic evaluation. This file will come in very handy this week when we need to start implementing training procedures for our agents. 

As part of this effort, I also created a series of tests that can be found in [`test_environment.py`](https://github.com/josemarquezjaramillo/crypto-rl-portfolio/blob/main/tests/test_environment.py)

3. **Base Agent class**: In order to have a unified framework, I created a baseclass agent that could be modified according to the different agent specifications in our project. The agents are to be stored in the [agents](https://github.com/josemarquezjaramillo/crypto-rl-portfolio/tree/main/agents) directory of the project. For this, I created `base_agent.py` which implements:

    - `MetricsTracker`: Lightweight tracker for agent performance metrics. Collects step-level data during episodes and computes summary
    statistics at episode end. Designed for minimal overhead.
    - `BaseAgent`: Abstract base class for portfolio management agents. Provides common infrastructure for training, evaluation, logging,
    and checkpointing. All agents (LinUCB, DQN, REINFORCE) extend this class and implement agent-specific methods. It follows what we have done previously in class projects. 

**Difficulties encountered and overcome:**

In the past, I have mostly developed these kinds of projects individually. It took significant consideration and effort developing a framework of data that we could share and use not only across agents but also across different computing machines and development environments. Considering that Taylor will develop using Google Colab, I hope that this setup is convenient for him. 

**What I learned this week:**

Before doing any coding this week, I had to think very carefully on how to specify our state space. I propose an initial version that is replicated in most of our reference papers (with some variations). In the end we will define a state space as the tuple (X_t,w_{t−1}):

    - For every tradable asset i at time t, we build a lookback window of length 60 calendar days. For each of those 60 days we collect four raw features:

        1. Close price
        2. High price
        3. Low price
        4. Volume

        We then stack those windows across assets, producing a tensor:

            X_t ∈ ℝ^{A_t × 4 × 60}
    
    - Normalization is implemented in two particular ways:
        1. Price normalization: Divide `close`, `high`, and `low` at each of the 60 lookback days by the asset’s close on day t. This rescales the series so that the most recent close is 1.0 and all prior prices are relative. This improves stationarity and was shown to stabilize training in crypto portfolio agents.
        2.  Volume normalization: Apply log(1+volume) for each day in the 60-day window. Z-score within that 60-day window (subtract mean, divide by standard deviation).

    - The agent's observation at time t also includes the previous realized portfolio allocation w_{t−1}, i.e. the weights we were actually holding going into day t. This vector is not derivable from OHLCV alone and must be provided by the environment. Conditioning on w_{t−1} is analogous to the "Portfolio Vector Memory" (PVM) mechanism in Jiang et. al (2017), which helps the policy learn to internalize turnover costs and not churn unnecessarily.

With regards to the data portion I learned that not all agents split data in the same way. For example, our reference paper in DQN (Lucarelli & Borrotti 2019) splits data for training, validation, and testing. They use the validation set for hyperparameter tuning. However, because cryptocurrencies have gone through several volatility regimes over the last several years, using a normal timeseries split is not representative of the actual data. They therefore sample the validation (hyperparameter tuning set) from specific periods. They then train the overall model using both the training and the validation set. In contrast, our reference REINFORCE+baseline (Jiang et. al 2017) does not. They only use a training and test set without any hyperparameter tuning. It is a similar case with our bandit reference studies (Fonseca et. al 2024 and Huo and Fu 2017). It may therefore be the case that each agent type will need to receive special considering during the training phase. 

This week reinforced the critical importance of reading original paper implementations, not just the papers themselves. By consulting Jiang's GitHub repository, I discovered implementation details (fixed coin counts, exact window handling) that weren't fully specified in the paper. I also learned the value of extensive design discussion before coding - answering those 6 architectural questions upfront saved significant refactoring time. The experience of balancing "simplicity and effectiveness over over-engineered complexity" taught me that pragmatic solutions (like hybrid backfilling) often work better than theoretically perfect but complex approaches. Finally, I gained deeper appreciation for proper Python package structure and the importance of using relative imports in multi-module projects.

---

## Teammate's Tasks & Accomplishments

Taylor has made solid progress on both the research and implementation fronts this week:

1. **Research & Planning**: Taylor has been researching the implementation details of the LinUCB and REINFORCE+baseline agents, reviewing reference papers and comparing different approaches. Considering feedback he received from last week's report, he has decided to prioritize the REINFORCE+baseline agent first, then move to the LinUCB agent.

2. **Development Environment Setup**: Taylor successfully configured a working environment on Google Colab that is compatible with our shared data infrastructure. He has been able to install dependencies and successfully run both the data loading modules and the portfolio environment code. This confirms that our shared dataset and environment design works across different development platforms.

3. **Agent Implementation - In Progress**: Taylor has begun implementing the policy gradient agent. This will serve as the foundation for the REINFORCE+baseline implementation, which is scheduled for completion this coming week.

**Next steps for Taylor**: Continue development of the REINFORCE+baseline agent, leveraging the `BaseAgent` class and the exported datasets from `dataset_v1/`. Once the policy gradient implementation is functional, he will begin training runs and hyperparameter exploration.

---

## GitHub Activity

**Repository:** https://github.com/josemarquezjaramillo/crypto-rl-portfolio

**Completed this week:**
- ✅ [Port existing data infrastructure](https://github.com/josemarquezjaramillo/crypto-rl-portfolio/issues/1)
- ✅ [Implement portfolio environment](https://github.com/josemarquezjaramillo/crypto-rl-portfolio/issues/2)
- ✅ [Create a BaseAgent class](https://github.com/josemarquezjaramillo/crypto-rl-portfolio/issues/13)

**Milestone progress:** Week 1: Data Infrastructure

---

## Risks, Concerns & Timeline Status

**Risks and concerns:**

No significant risks at this time. Next week I will be overseas and my time available to work on the project during the weekend will likely be limited. I will therefore do my best to work on my tasks during the week. 

**On track to finish on time:**

So far I have met all of my planned tasks.

---

## Key References

This week's work drew primarily from the following papers:

- **Jiang et al. (2017)** - "A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem" - Used for state space design (60-day lookback window, Portfolio Vector Memory mechanism), normalization strategy, and environment architecture. [[PDF](https://github.com/josemarquezjaramillo/crypto-rl-portfolio/blob/main/reference_papers/jiang2017eIIE%20-%20A%20Deep%20Reinforcement%20Learning%20Framework%20for%20the%20Financial%20Portfolio%20Management%20Problem.pdf)]

- **Lucarelli & Borrotti (2020)** - "Deep Reinforcement Learning for Cryptocurrency Trading" - Referenced for DQN-specific implementation details, data splitting strategy (train/validation/test), and validation set sampling approach for different volatility regimes. [[PDF](https://github.com/josemarquezjaramillo/crypto-rl-portfolio/blob/main/reference_papers/lucarelli2020dqlcrypto%20-%20Deep%20Reinforcement%20Learning%20for%20Cryptocurrency%20Trading.pdf)]

- **Jiang et al. (2016)** - "Cryptocurrency Portfolio Management with Deep Reinforcement Learning" - Consulted for data quality handling, gap-filling strategies, and survivorship bias considerations. [[PDF](https://github.com/josemarquezjaramillo/crypto-rl-portfolio/blob/main/reference_papers/jiang2016drlt%20-%20Cryptocurrency%20Portfolio%20Management%20with%20Deep%20Reinforcement%20Learning..pdf )]

- **Fonseca et al. (2024)** - "Improving Portfolio Optimization Results with Bandit Networks" - Reviewed for contextual bandit approach and training methodology comparison. [[PDF](https://github.com/josemarquezjaramillo/crypto-rl-portfolio/blob/main/reference_papers/fonseca2024banditnets%20-%20Improving%20Portfolio%20Optimization%20Results%20with%20Bandit%20Networks.pdf)]

- **Huo & Fu (2017)** - "Risk-aware Multi-armed Bandit and Portfolio Selection" - Consulted for LinUCB baseline implementation strategy. [[PDF](https://github.com/josemarquezjaramillo/crypto-rl-portfolio/blob/main/reference_papers/huo2017riskbandit%20-%20Risk-aware%20multi-armed%20bandit%20and%20portfolio%20selection.pdf)] 
