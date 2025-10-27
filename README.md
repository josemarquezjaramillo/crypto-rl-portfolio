# Reinforcement Learning for Cryptocurrency Portfolio Management

Deep reinforcement learning framework for dynamic cryptocurrency portfolio optimization with weekly rebalancing and transaction cost modeling.

## Overview

This project implements three RL algorithms for managing a long-only cryptocurrency portfolio:
- **LinUCB** (Contextual Bandit)
- **DQN** (Value-Based Deep Q-Network)
- **REINFORCE** (Policy Gradient)

Agents are trained and evaluated against classical baselines (Equal-Weight, Market-Cap, Mean-Variance) using real market data and Binance transaction costs.

## Key Features

- Weekly rebalancing with daily market observations
- Transaction cost modeling (0.10% per trade)
- Monthly universe reconstitution (top-10 cryptocurrencies by market cap)
- Cost-aware reward: log-return minus proportional trading costs
- Train/validation/test splits: 2019-2021 / 2022 / 2023-2025

## Project Structure
```
crypto-rl-portfolio/
├── data/               # Data loading and preprocessing
├── environment/        # MDP environment implementation
├── agents/            # RL agent implementations
├── baselines/         # Classical portfolio strategies
├── training/          # Training loops and hyperparameters
├── evaluation/        # Performance metrics and visualization
├── utils/             # Utility functions
├── tests/             # Unit tests
├── notebooks/         # Exploratory analysis
└── configs/           # Configuration files
```

## Installation
```bash
git clone https://github.com/YOUR-USERNAME/crypto-rl-portfolio.git
cd crypto-rl-portfolio
pip install -r requirements.txt
```

## Usage

Coming soon...

## Authors

Jose Márquez Jaramillo & Taylor Hawks  
Johns Hopkins University  
Fall 2025

## License

MIT License
